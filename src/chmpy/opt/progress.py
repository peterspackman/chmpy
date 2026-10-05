"""Progress reporting for the long-running calculations.

Every function here that can take a while -- `relax`, `elastic_tensor`,
`force_constants`, `lattice_energy` -- takes one argument for it, `progress`:

* `None` (the default) or `False`: silent;
* `True`: print a readable line per event, indented by how deeply it is nested;
* a callable: called with a `Progress` for every event.

    result = calc.lattice_energy(crystal, progress=True)

    def bar(event):
        if event.depth == 0 and event.total:
            print(f"{event.index + 1}/{event.total} {event.stage}")

    elastic_tensor(crystal, calc, progress=bar)

Events nest. A lattice energy is a crystal relaxation followed by one per
molecule, and each relaxation reports its own steps one level further down, so
a hook that only wants the outline can ignore everything with `depth > 0`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Progress:
    """One thing that happened.

    Attributes:
        task: what is running: "relax", "elastic", "force constants",
            "lattice energy"
        stage: which part of it: "crystal", "molecule 1", "voigt 3 +0.01",
            "atom 0 along [1, 0, 0]", "step"
        index: position of this stage among `total`, from zero, when known
        total: how many such stages there are, when known. For a relaxation
            this is the step cap, which the run may well finish under.
        message: the line `progress=True` prints
        depth: how deeply nested the task is; zero for the call the user made
        parents: the `(task, stage)` of each enclosing event, outermost first
        step: the optimiser's `Step`, for an event from inside a relaxation
        done: whether this event reports a stage finishing rather than starting
    """

    task: str
    stage: str
    message: str
    index: int | None = None
    total: int | None = None
    depth: int = 0
    parents: tuple = ()
    step: object = None
    done: bool = False


def print_progress(event: Progress, stream=None) -> None:
    """The hook `progress=True` installs: one indented line per event."""
    stream = stream if stream is not None else sys.stdout
    counter = (
        f"[{event.index + 1}/{event.total}] "
        if event.index is not None and event.total and event.step is None
        else ""
    )
    print(f"{'  ' * event.depth}{counter}{event.message}", file=stream, flush=True)


@dataclass(frozen=True)
class Reporter:
    """Where events go, and the context they are nested in.

    Built from the user's `progress` argument with `reporter`, and handed down
    with `nested` so that the events from an inner calculation say what they
    are part of.
    """

    hook: object = None
    depth: int = 0
    parents: tuple = field(default=())

    @property
    def active(self) -> bool:
        return self.hook is not None

    def __call__(
        self,
        task,
        stage,
        message,
        index=None,
        total=None,
        step=None,
        done=False,
    ) -> None:
        if self.hook is None:
            return
        self.hook(
            Progress(
                task=task,
                stage=stage,
                message=message,
                index=index,
                total=total,
                depth=self.depth,
                parents=self.parents,
                step=step,
                done=done,
            )
        )

    def nested(self, task, stage) -> Reporter:
        "The reporter for work done inside `(task, stage)`"
        if self.hook is None:
            return self
        return Reporter(self.hook, self.depth + 1, (*self.parents, (task, stage)))


#: the reporter that reports nothing
SILENT = Reporter()


def reporter(progress) -> Reporter:
    """Turn a `progress` argument into a `Reporter`.

    Args:
        progress: None or False for silence, True to print, a callable taking
            a `Progress`, or a `Reporter` already (when one calculation hands
            its context to another)
    """
    if isinstance(progress, Reporter):
        return progress
    if progress is None or progress is False:
        return SILENT
    if progress is True:
        return Reporter(print_progress)
    if callable(progress):
        return Reporter(progress)
    raise TypeError(
        f"progress should be True, False, None or a callable, not {progress!r}"
    )
