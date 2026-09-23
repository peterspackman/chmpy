"""Metatomic models, driven directly.

`metatomic` ships an ASE calculator that `Calculator.from_ase` will wrap. This
backend talks to the model instead, which gains three things:

* the model's inputs -- atomic types, positions, cell, neighbour lists -- come
  straight from a `System`, with no intermediate container;
* neighbour lists are the skinned, cached ones the rest of `chmpy.calc` uses,
  so a small optimiser step reuses the pair enumeration;
* `compute_batch` puts every system into one call of the model and one backward
  pass, which is most of what a GPU offers when the cells are small.

Measured on CPU with four-atom cells, batching forty systems is 2.6 times
faster than evaluating them one at a time; the container itself is worth about
three percent.

Units are metatomic's own -- eV and Angstroms -- so nothing is converted.
"""

from __future__ import annotations

import numpy as np

from chmpy.util.optional import require

from ..base import Calculator, PropertyNotAvailable
from ..neighbors import NeighborList
from ..result import ENERGIES, FORCES, STRESS, Result

#: extra range enumerated in the neighbour lists, as a fraction of the cutoff,
#: so that small optimiser steps reuse the pair list
DEFAULT_SKIN = 0.1


class MetatomicCalculator(Calculator):
    """A metatomic `AtomisticModel`, evaluated without going through ASE.

    Args:
        model: a path to an exported model, or an already-loaded one
        device: torch device, e.g. "cpu", "cuda", "mps". Defaults to the first
            device the model says it supports.
        dtype: torch dtype. Defaults to the model's own.
        skin: neighbour-list skin as a fraction of each requested cutoff
        check_consistency: pass metatomic's internal consistency checks
        extensions_directory: where to find TorchScript extensions, if the
            model needs them

    Examples:
        Relaxing a crystal with PET-MAD::

            calc = MetatomicCalculator("pet-mad-latest.pt")
            result = relax(crystal, calc, fmax=0.01, smax=0.05)
    """

    provides = {"energy", "forces", "stress", "energies"}

    def __init__(
        self,
        model,
        device=None,
        dtype=None,
        skin: float = DEFAULT_SKIN,
        check_consistency: bool = False,
        extensions_directory=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.torch = require("torch", "running a metatomic model")
        self.metatomic = require("metatomic.torch", "running a metatomic model")

        self.model = self._load(model, extensions_directory)
        self.check_consistency = check_consistency
        capabilities = self.model.capabilities()
        self.energy_key = _energy_key(capabilities)

        self.device = self.torch.device(
            _pick_device(self.metatomic, capabilities.supported_devices, device)
        )
        self.dtype = dtype or getattr(self.torch, capabilities.dtype)
        self.model = self.model.to(self.device)
        # the model states its own precision, so there is nothing to infer: a
        # float32 model cannot resolve energy differences below about 1e-7
        # relative, and `chmpy.opt.TrustRegion` needs to know that to avoid
        # reading rounding noise as "the step achieved nothing"
        self.energy_precision = float(
            np.finfo(np.float32 if self.dtype == self.torch.float32 else np.float64).eps
        )

        self._neighbor_lists = [
            (
                options,
                NeighborList(
                    cutoff=options.engine_cutoff(engine_length_unit="angstrom"),
                    full=options.full_list,
                    skin=skin * options.engine_cutoff(engine_length_unit="angstrom"),
                ),
            )
            for options in self.model.requested_neighbor_lists()
        ]

    def _load(self, model, extensions_directory):
        if isinstance(model, str):
            return self.metatomic.load_atomistic_model(
                model, extensions_directory=extensions_directory
            )
        return model

    def __repr__(self) -> str:
        return f"<MetatomicCalculator device={self.device} dtype={self.dtype}>"

    # -- evaluation ----------------------------------------------------------

    def compute(self, system, want):
        return self.compute_batch([system], want)[0]

    def compute_batch(self, systems, want):
        """Every system in one call of the model, and one backward pass.

        The energies of the batch are summed before the backward, which is
        valid because each system's gradient only reaches its own positions and
        its own strain -- there is no term in the model coupling one structure
        to another.
        """
        need_gradients = bool(want & {FORCES, STRESS})

        prepared = [self._prepare(system, need_gradients) for system in systems]
        outputs = self._run([entry["system"] for entry in prepared], want)

        energies = outputs[self.energy_key]
        per_atom = ENERGIES in want
        block = energies.block()
        if per_atom:
            import metatensor.torch as mts

            total = mts.sum_over_samples(energies, sample_names=["atom"]).block().values
        else:
            total = block.values

        if need_gradients:
            total.sum().backward()

        results = []
        for index, entry in enumerate(prepared):
            results.append(self._unpack(entry, total, block, index, want, per_atom))
        return results

    def _run(self, systems, want):
        metatomic = self.metatomic
        output = metatomic.ModelOutput(
            quantity="energy",
            unit="eV",
            per_atom=ENERGIES in want,
            explicit_gradients=[],
        )
        options = metatomic.ModelEvaluationOptions(
            length_unit="angstrom",
            outputs={self.energy_key: output},
            selected_atoms=None,
        )
        return self.model(systems, options, check_consistency=self.check_consistency)

    def _prepare(self, system, need_gradients):
        """One metatomic System, with its neighbour lists and autograd handles."""
        torch = self.torch
        metatomic = self.metatomic

        # torch.tensor rather than from_numpy: a System's arrays are read-only,
        # and torch warns (loudly, once) about wrapping a non-writable array
        types = torch.tensor(
            np.asarray(system.numbers), dtype=torch.int32, device=self.device
        )
        positions = torch.tensor(
            np.asarray(system.positions), dtype=self.dtype, device=self.device
        )
        cell = torch.tensor(
            np.asarray(system.cell), dtype=self.dtype, device=self.device
        )
        pbc = torch.tensor(np.asarray(system.pbc), dtype=torch.bool, device=self.device)

        strain = None
        if need_gradients:
            positions.requires_grad_(True)
            strain = torch.eye(
                3, requires_grad=True, device=self.device, dtype=self.dtype
            )
            # r -> r (I + eps) and A -> A (I + eps): the same strain the stress
            # is the derivative with respect to
            strained = positions @ strain
            strained.retain_grad()
            cell = cell @ strain
        else:
            strained = positions

        metatomic_system = metatomic.System(types, strained, cell, pbc)
        for options, neighbor_list in self._neighbor_lists:
            block = self._neighbor_block(system, neighbor_list)
            metatomic.register_autograd_neighbors(
                metatomic_system, block, check_consistency=self.check_consistency
            )
            metatomic_system.add_neighbor_list(options, block)

        return {
            "system": metatomic_system,
            "strain": strain,
            "positions": strained,
            "volume": system.volume,
            "n_atoms": len(system),
        }

    def _neighbor_block(self, system, neighbor_list):
        """Our pair list, in the layout metatomic wants."""
        import metatensor.torch as mts

        torch = self.torch
        pairs = neighbor_list.compute(system)
        samples = np.concatenate(
            [
                pairs.i[:, None].astype(np.int32),
                pairs.j[:, None].astype(np.int32),
                pairs.shifts.astype(np.int32),
            ],
            axis=1,
        )
        return mts.TensorBlock(
            values=torch.tensor(
                pairs.vectors, dtype=self.dtype, device=self.device
            ).reshape(-1, 3, 1),
            samples=mts.Labels(
                names=[
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                values=torch.tensor(samples, dtype=torch.int32, device=self.device),
                assume_unique=True,
            ),
            components=[mts.Labels.range("xyz", 3).to(self.device)],
            properties=mts.Labels.range("distance", 1).to(self.device),
        )

    def _unpack(self, entry, total, block, index, want, per_atom):
        torch = self.torch
        energy = float(total[index].detach().to("cpu").to(torch.float64).item())

        forces = stress = energies = None
        if FORCES in want:
            gradient = entry["positions"].grad
            if gradient is None:
                raise PropertyNotAvailable(
                    "the model produced an energy that does not depend on the "
                    "positions, so it has no forces"
                )
            forces = -gradient.detach().to("cpu").to(torch.float64).numpy()
        if STRESS in want:
            gradient = entry["strain"].grad
            if gradient is None:
                raise PropertyNotAvailable(
                    "the model produced an energy that does not depend on the "
                    "cell, so it has no stress"
                )
            stress = (
                gradient.detach().to("cpu").to(torch.float64).numpy() / entry["volume"]
            )
            stress = 0.5 * (stress + stress.T)
        if per_atom:
            samples = block.samples
            mask = samples.column("system") == index
            values = block.values[mask].detach().reshape(-1)
            energies = values.to("cpu").to(torch.float64).numpy()

        return Result(
            energy=energy,
            forces=forces,
            stress=stress,
            energies=energies,
            volume=entry["volume"],
        )


def _pick_device(metatomic, supported, desired):
    """The best device the model supports and this machine actually has.

    A model's `supported_devices` is a preference order, not an availability
    list: PET-MAD names "cuda" first whether or not this torch was built with
    it, so taking the first entry fails on a laptop.
    """
    picker = getattr(metatomic, "pick_device", None)
    if picker is not None:
        return picker(list(supported), desired)
    import torch

    available = {"cpu": True, "cuda": torch.cuda.is_available()}
    available["mps"] = getattr(torch.backends, "mps", None) is not None and (
        torch.backends.mps.is_available()
    )
    if desired and available.get(desired.split(":")[0], True):
        return desired
    for name in supported:
        if available.get(name, False):
            return name
    return "cpu"


def _energy_key(capabilities) -> str:
    """Which of the model's outputs is the energy."""
    outputs = list(capabilities.outputs.keys())
    if "energy" in outputs:
        return "energy"
    candidates = [name for name in outputs if "energy" in name]
    if not candidates:
        raise PropertyNotAvailable(
            f"this model has no energy output; it produces {outputs}"
        )
    return candidates[0]


def pet_mad(version: str = "latest", **kwargs) -> MetatomicCalculator:
    """The PET-MAD model, as a calculator.

    Args:
        version: the PET-MAD version to load
        **kwargs: passed to `MetatomicCalculator`

    Returns:
        MetatomicCalculator
    """
    pet = require("pet_mad.calculator", "loading PET-MAD")
    return MetatomicCalculator(pet.PETMADCalculator(version=version)._model, **kwargs)
