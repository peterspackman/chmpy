"""The quadratic model a trust region minimises.

The optimiser needs three things of it: `B v` for the predicted reduction and
the Cauchy point, `B^-1 g` for the Newton step, and an update from a step and
the change in gradient it produced. The dogleg, the accept/reject test and the
radius policy are written against those, so another model can be supplied by
passing `model=` to `TrustRegion`.

`B0`, the model before any curvature pairs, may be a full matrix rather than a
diagonal. It never changes, so it is factorised once at construction and every
later solve is a triangular pair -- `O(n^3)` once rather than per iteration.
That is what lets a model Hessian contribute its couplings and not merely its
diagonal, and the difference is large: with only the diagonal this model needed
55 calculator calls on a relaxation where a full BFGS needed 51, and with the
couplings it needs 23. Above `FULL_MODEL_LIMIT` freedoms the starting model is
too large to hold and factorise, so only its diagonal survives.

Keeping `B0` fixed and applying the pairs as a correction is also why this is
the only model here. A full BFGS accumulates its updates into the starting
matrix, so a model Hessian is progressively overwritten by the early, noisy
curvature pairs; here it preconditions the whole run, and what gets forgotten
is the stale pairs instead.

The update is Powell-damped, so the model stays positive definite for any
curvature pair and the two-loop inverse can be trusted without a fallback.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, get_blas_funcs, lu_factor, lu_solve

#: how many curvature pairs a limited-memory model keeps
DEFAULT_MEMORY = 12

#: above this many degrees of freedom a starting model is no longer kept whole:
#: holding and factorising it costs `n^2` and `O(n^3)`, so only its diagonal is
#: carried into the limited-memory model
FULL_MODEL_LIMIT = 1000


def damped_pair(step, gradient_change, model_step, curvature):
    """Powell's damping: blend `y` towards `Bs` until the pair is usable.

    Returns `(y_hat, y_hat . s)`, or `(None, 0)` when no update is safe. The
    blend is chosen so that `y_hat . s` comes out at exactly `0.2 s.Bs > 0`,
    which keeps the updated model positive definite for any step rather than
    only for those with positive curvature. Skipping instead -- which plain
    BFGS must do -- throws away precisely the steps a trust region rejects, so
    the model that caused a rejection never learns from it.
    """
    along = float(gradient_change @ step)
    if along >= 0.2 * curvature:
        adjusted = gradient_change
    else:
        theta = 0.8 * curvature / (curvature - along)
        adjusted = theta * gradient_change + (1.0 - theta) * model_step
    denominator = float(adjusted @ step)
    if not denominator > 1e-30:
        return None, 0.0
    return adjusted, denominator


class LimitedMemoryBFGS:
    """The last `memory` curvature pairs, and a diagonal starting model.

    The inverse comes from the usual two-loop recursion. The forward product
    comes from the compact representation of Byrd, Nocedal and Schnabel,

        B = B0 - [B0 S, Y] M^-1 [S^T B0; Y^T],
        M = [[S^T B0 S, L], [L^T, -D]]

    which is exact rather than an approximation of the stored model, and costs
    `O(n m + m^3)` with `m` around ten. It is needed because a dogleg wants
    `g.B.g` and `p.B.p`, not only the Newton step.

    Args:
        n: number of degrees of freedom
        memory: how many curvature pairs to keep
        diagonal: (n,) starting curvature per degree of freedom. The diagonal
            of a model Hessian goes here, which keeps the part of it that a
            limited-memory model can carry -- which freedoms are stiff -- while
            dropping the couplings it cannot.
    """

    def __init__(
        self, n: int, memory: int = DEFAULT_MEMORY, diagonal=None, initial=None
    ):
        self.n = int(n)
        self.memory = int(memory)
        self._matrix = None
        self._factor = None
        if initial is not None:
            # A full starting model. B0 never changes, so it is factorised once
            # here and every later solve is a triangular pair -- O(n^3) at
            # construction rather than per iteration, which is the whole
            # difference from a dense model. This is what lets a limited-memory
            # model start from a model Hessian's *couplings* rather than only
            # its diagonal.
            matrix = np.asarray(initial, dtype=float)
            scale = float(np.mean(np.diag(matrix)))
            scale = scale if scale > 0 else 1.0
            try:
                self._matrix = np.asfortranarray(matrix / scale)
                self._factor = cho_factor(self._matrix, lower=True)
            except np.linalg.LinAlgError:
                # A starting model that is not positive definite -- an analytic
                # Hessian at a saddle, say. Its couplings cannot be used as a
                # B0, since B0 has to be invertible in a known direction, so
                # fall back to its diagonal with the negative entries floored.
                self._matrix = self._factor = None
                initial = None
            else:
                self._symv = get_blas_funcs("symv", (self._matrix,))
                shape = np.diag(self._matrix)
        if initial is None:
            shape = np.ones(n) if diagonal is None else np.asarray(diagonal, float)
            shape = np.where(shape > 0, shape, 1.0)
            scale = float(np.mean(shape))
            shape = shape / scale
        # The model is split into a shape (mean diagonal one) and a magnitude,
        # so the magnitude can follow the most recent curvature without
        # disturbing the anisotropy a model Hessian supplied. The magnitude
        # starts at the one it came with: normalising the shape and then
        # starting gamma at 1 threw the starting model's scale away, so an
        # exactly correct starting Hessian did not give the exact Newton step.
        self._shape = shape
        self._initial_gamma = scale if scale > 0 else 1.0
        self.reset()

    def reset(self) -> None:
        self.steps: list[np.ndarray] = []
        self.changes: list[np.ndarray] = []
        self.gamma = self._initial_gamma
        self._compact_cache = None

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def diagonal(self) -> np.ndarray:
        "The diagonal of `B0`, for a model that has no couplings to offer"
        return self.gamma * self._shape

    def _base_matvec(self, vector) -> np.ndarray:
        """`B0 v`.

        One definition, used by both `matvec` and `solve`. They described
        different models when the scalar scaling was applied to only one of
        them, which makes a dogleg take a Newton step from one model and score
        it against another.
        """
        if self._matrix is None:
            return self.diagonal * vector
        return self.gamma * self._symv(1.0, self._matrix, vector)

    def _base_solve(self, vector) -> np.ndarray:
        "`B0^-1 v`, from the factorisation taken once at construction"
        if self._matrix is None:
            return vector / self.diagonal
        return cho_solve(self._factor, vector) / self.gamma

    # -- the model -----------------------------------------------------------

    def matvec(self, vector) -> np.ndarray:
        base = self._base_matvec(vector)
        if not self.steps:
            return base
        columns, factor = self._compact()
        return base - columns @ lu_solve(factor, columns.T @ vector)

    def solve(self, vector) -> np.ndarray:
        """`B^-1 v` by the two-loop recursion, unwinding through the same `B0`."""
        q = np.array(vector, dtype=float)
        alphas, rhos = [], []
        for step, change in zip(
            reversed(self.steps), reversed(self.changes), strict=True
        ):
            rho = 1.0 / float(change @ step)
            alpha = rho * float(step @ q)
            q -= alpha * change
            alphas.append(alpha)
            rhos.append(rho)

        q = self._base_solve(q)
        for (step, change), alpha, rho in zip(
            zip(self.steps, self.changes, strict=True),
            reversed(alphas),
            reversed(rhos),
            strict=True,
        ):
            beta = rho * float(change @ q)
            q += (alpha - beta) * step
        return q

    def update(self, step, gradient_change) -> bool:
        model_step = self.matvec(step)
        curvature = float(step @ model_step)
        if not curvature > 0:
            return False
        adjusted, denominator = damped_pair(
            step, gradient_change, model_step, curvature
        )
        if adjusted is None:
            return False
        self.steps.append(np.array(step, dtype=float))
        self.changes.append(np.array(adjusted, dtype=float))
        if len(self.steps) > self.memory:
            self.steps.pop(0)
            self.changes.pop(0)
        # the usual scalar scaling of B0, on top of the supplied shape
        self.gamma = float(adjusted @ adjusted) / denominator
        self._compact_cache = None
        return True

    def rescale(self, old_scale, new_scale) -> None:
        ratio = np.asarray(old_scale) / np.asarray(new_scale)
        self.steps = [step / ratio for step in self.steps]
        self.changes = [change * ratio for change in self.changes]
        if self._matrix is not None:
            outer = np.outer(ratio, ratio)
            self._matrix = np.asfortranarray(self._matrix * outer)
            self._factor = cho_factor(self._matrix, lower=True)
            self._shape = np.diag(self._matrix)
        else:
            self._shape = self._shape * ratio * ratio
            self._shape = self._shape / float(np.mean(self._shape))
        self._compact_cache = None

    def quadratic(self, gradient, step) -> float:
        "The model's predicted reduction, `-(g.p + p.B.p / 2)`"
        return -(float(gradient @ step) + 0.5 * float(step @ self.matvec(step)))

    def __repr__(self) -> str:
        return f"<LimitedMemoryBFGS {self.n} dof, {len(self)} pairs>"

    def to_dense(self) -> np.ndarray:
        """The model as an explicit matrix, by applying it to a basis.

        `O(n^2 m)`, so only worth asking for at the sizes where a dense model
        would have been an option at all. Used to hand what one stage of a
        relaxation learned to the next.
        """
        return np.column_stack([self.matvec(column) for column in np.eye(self.n)])

    def _compact(self):
        """`[B0 S, Y]` and an LU factorisation of the middle matrix.

        Cached until the model changes. Rebuilding it inside `matvec` -- which
        a dogleg calls three times an iteration, for the Cauchy point and twice
        for the predicted reduction -- meant re-stacking the stored pairs and
        re-factorising a 2m x 2m matrix each time, for no new information.
        """
        if self._compact_cache is None:
            s = np.stack(self.steps, axis=1)
            y = np.stack(self.changes, axis=1)
            scaled = np.column_stack([self._base_matvec(column) for column in s.T])
            sy = s.T @ y
            lower = np.tril(sy, -1)
            middle = np.block([[s.T @ scaled, lower], [lower.T, -np.diag(np.diag(sy))]])
            columns = np.concatenate([scaled, y], axis=1)
            self._compact_cache = (columns, lu_factor(middle))
        return self._compact_cache


def for_size(n_dof: int, initial=None, memory: int = DEFAULT_MEMORY):
    """Build a curvature model, handling the starting model by size.

    Args:
        n_dof: number of degrees of freedom
        initial: (n, n) starting model, kept whole while it is small enough to
            factorise once and reduced to its diagonal beyond that
        memory: curvature pairs to keep

    Returns:
        LimitedMemoryBFGS
    """
    if initial is not None:
        if n_dof <= FULL_MODEL_LIMIT:
            return LimitedMemoryBFGS(n_dof, memory=memory, initial=initial)
        diagonal = np.diag(np.asarray(initial)).copy()
        return LimitedMemoryBFGS(n_dof, memory=memory, diagonal=diagonal)
    return LimitedMemoryBFGS(n_dof, memory=memory)
