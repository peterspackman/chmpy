BOHR_TO_ANGSTROM = 0.52917749
AU_TO_KJ_PER_MOL = 2625.499639


class units:
    factors = {
        ("angstrom", "angstrom"): 1,
        ("au", "au"): 1,
        ("au", "bohr"): 1,
        ("bohr", "au"): 1,
        ("bohr", "bohr"): 1,
        ("kj_per_mol", "kj_per_mol"): 1,
        ("bohr", "angstrom"): BOHR_TO_ANGSTROM,
        ("au", "angstrom"): BOHR_TO_ANGSTROM,
        ("angstrom", "bohr"): 1 / BOHR_TO_ANGSTROM,
        ("angstrom", "au"): 1 / BOHR_TO_ANGSTROM,
        ("au", "kj_per_mol"): AU_TO_KJ_PER_MOL,
        ("hartree", "kj_per_mol"): AU_TO_KJ_PER_MOL,
        ("kj_per_mol", "au"): 1 / AU_TO_KJ_PER_MOL,
        ("kj_per_mol", "hartree"): 1 / AU_TO_KJ_PER_MOL,
    }

    @classmethod
    def _s_unit(cls, unit):
        return unit.lower().replace("/", "_per_")

    @classmethod
    def _conversion_factor(cls, f, t):
        if (f, t) in cls.factors:
            return cls.factors[(f, t)]
        raise ValueError(f"No viable conversion from '{f}' to '{t}'")

    @classmethod
    def convert(cls, value, t="au", f="au"):
        tu = cls._s_unit(t)
        try:
            return getattr(cls, tu)(value, unit=f)
        except AttributeError as e:
            raise ValueError(f"Unknown unit {t}") from e

    @classmethod
    def bohr(cls, value, unit="au"):
        unit = cls._s_unit(unit)
        return value * cls._conversion_factors(unit, "bohr")

    @classmethod
    def angstrom(cls, value, unit="au"):
        unit = cls._s_unit(unit)
        return value * cls._conversion_factor(unit, "angstrom")

    @classmethod
    def au(cls, value, unit="au"):
        unit = cls._s_unit(unit)
        return value * cls._conversion_factor(unit, "au")

    @classmethod
    def kj_per_mol(cls, value, unit="au"):
        unit = cls._s_unit(unit)
        return value * cls._conversion_factor(unit, "kj_per_mol")
