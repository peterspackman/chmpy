"""PET models (PET-MAD etc.) through pet-kokkos, with no torch.

pet-kokkos is a C++/Kokkos implementation of the PET architecture that runs
the published UPET models on CPUs and on NVIDIA and AMD GPUs. It returns the
energy, per-atom energies, forces and virial in one call, and does its own
neighbour search, so there is nothing for this adapter to supply but the
geometry.

    from chmpy.calc.adapters.pet_kokkos import PetKokkosCalculator

    calc = PetKokkosCalculator("pet-mad-s")
    result = relax(crystal, calc, fmax=0.01, smax=0.05)

The CPU build is on PyPI (`pip install 'chmpy[pet]'`). GPU builds are the same
package with a local version label, served from their own index:

    uv pip install pet-kokkos --extra-index-url https://peterspackman.github.io/pet-kokkos/cu12

`pet_kokkos.backend()` says which one is installed.

Units are the model's -- eV and Angstroms -- so nothing is converted. The
virial pet-kokkos returns is `+dE/deps` in Voigt order xx, yy, zz, xy, xz, yz,
so the stress is that over the volume.
"""

from __future__ import annotations

import numpy as np

from chmpy.util.optional import require

from ..base import Calculator, PropertyNotAvailable
from ..result import FORCES, STRESS, Result

#: pet-kokkos' Voigt order, xx yy zz xy xz yz, as (row, column) pairs
_VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

#: relative energy noise for each GEMM/attention mode, for `energy_noise`.
#: Measured as the scatter about a smooth fit along a 0.02 A scan of 54-atom
#: diamond with pet-mad-s on an RTX 4080: fp32 gives 5e-9 (well inside float32
#: epsilon, which is used as the bound), tf32 3e-6 with a worst case of 8e-6.
#: fp16 matches tf32's accuracy in the README; bf16 has 8x coarser mantissa.
_PRECISION = {
    "native": float(np.finfo(np.float32).eps),
    "tf32": 1e-5,
    "fp16": 1e-5,
    "bf16": 1e-4,
    "fp8": 1e-3,
}


class PetKokkosCalculator(Calculator):
    """A PET model evaluated by pet-kokkos.

    Args:
        model: a published model's name, e.g. "pet-mad-s", "pet-mad-xs",
            "pet-omat-l" (downloaded on first use), or a path prefix to
            `<prefix>.json` and `<prefix>.safetensors`
        gemm: "native" (fp32), "tf32", "fp16", ... for the dense layers. This
            is process-wide in pet-kokkos, so it affects every calculator in
            the process; None leaves it as it is.
        attention: "native" or "tf32" for the attention products, likewise
            process-wide
        cache_neighbors: keep pet-kokkos' Verlet cache of the neighbour
            topology between calls, which is what a relaxation or a finite
            difference wants. It is checked on every call, so changing
            structure underneath it is safe.
        cache_size, fd_step: as for `Calculator`
        **options: passed to `pet_kokkos.Calculator`, e.g. `recompute`,
            `memory_budget_gib`, `max_batch_atoms`, `device_neighbors`

    A charge and spin multiplicity, for the models conditioned on them
    (pet-omol), are read from `system.info["charge"]` and
    `system.info["spin_multiplicity"]` (or `"spin"`, as ASE users write it).
    """

    provides = {"energy", "forces", "stress", "energies"}

    def __init__(
        self,
        model: str = "pet-mad-s",
        *,
        gemm: str | None = None,
        attention: str | None = None,
        cache_neighbors: bool = True,
        cache_size: int | None = None,
        fd_step: float | None = None,
        **options,
    ):
        super().__init__(cache_size=cache_size, fd_step=fd_step)
        pet_kokkos = require("pet_kokkos", "running a PET model with pet-kokkos")
        self.model_name = model
        self.pet = pet_kokkos.Calculator(
            model,
            gemm=gemm,
            attention=attention,
            cache_neighbors=cache_neighbors,
            **options,
        )
        self.backend = pet_kokkos.backend()
        self.cutoff = float(self.pet.cutoff)
        self.atomic_types = frozenset(int(z) for z in self.pet.atomic_types)
        # The network runs in fp32 and only the sums over atoms are fp64, so
        # the total is not itself a float32 value and the inference in
        # `Calculator.energy_noise` would wrongly conclude it is exact.
        self.energy_precision = max(
            _PRECISION.get(gemm or "native", _PRECISION["native"]),
            _PRECISION.get(attention or "native", _PRECISION["native"]),
        )

    def __repr__(self) -> str:
        return f"<PetKokkosCalculator {self.model_name} backend={self.backend}>"

    def close(self) -> None:
        "Free the model and its device memory now, rather than at exit"
        self.pet.close()

    # -- evaluation ----------------------------------------------------------

    def compute(self, system, want):
        self._check_types(system)
        cell, pbc = _cell_and_pbc(system)
        charge, multiplicity = _charge_and_spin(system)
        raw = self.pet.compute(
            system.numbers,
            system.positions,
            cell,
            pbc,
            forces=bool(want & {FORCES, STRESS}),
            charge=charge,
            spin_multiplicity=multiplicity,
        )
        return _unpack(raw, system, want)

    def compute_batch(self, systems, want):
        """Every system in one evaluation."""
        items = []
        for system in systems:
            self._check_types(system)
            cell, pbc = _cell_and_pbc(system)
            charge, multiplicity = _charge_and_spin(system)
            items.append(
                (system.numbers, system.positions, cell, pbc, charge, multiplicity)
            )
        raw = self.pet.compute_batch(items, forces=bool(want & {FORCES, STRESS}))
        return [
            _unpack(entry, system, want)
            for entry, system in zip(raw, systems, strict=True)
        ]

    def _check_types(self, system):
        """Name the element the model has never seen, before pet-kokkos does."""
        unknown = {int(z) for z in np.unique(system.numbers)} - self.atomic_types
        if unknown:
            raise PropertyNotAvailable(
                f"{self.model_name} has no parameters for atomic number"
                f"{'s' if len(unknown) > 1 else ''} {sorted(unknown)}"
            )


def _cell_and_pbc(system):
    pbc = tuple(bool(p) for p in system.pbc)
    cell = np.asarray(system.cell) if any(pbc) else None
    return cell, pbc


def _charge_and_spin(system):
    info = system.info
    charge = int(info.get("charge", 0))
    multiplicity = int(info.get("spin_multiplicity", info.get("spin", 1)))
    return charge, multiplicity


def _unpack(raw, system, want) -> Result:
    forces = stress = None
    if FORCES in want:
        forces = np.array(raw["forces"], dtype=np.float64)
    if STRESS in want:
        if "virial" not in raw:
            raise PropertyNotAvailable("pet-kokkos returned no virial")
        dE_deps = np.zeros((3, 3))
        for value, (i, j) in zip(raw["virial"], _VOIGT, strict=True):
            dE_deps[i, j] = dE_deps[j, i] = value
        stress = dE_deps / system.volume
    return Result(
        energy=float(raw["energy"]),
        forces=forces,
        stress=stress,
        energies=np.array(raw["energies"], dtype=np.float64),
        volume=system.volume,
    )
