BOHR_TO_ANGSTROM = 0.52917749
ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM
AU_TO_KJ_PER_MOL = 2625.499639
AU_TO_PER_CM = 219474.63
AU_TO_KCAL_PER_MOL = 627.5096080305927
KJ_TO_KCAL = 0.239006
EV_TO_KJ_PER_MOL = 96.48530749925973
AU_TO_EV = 27.211399
AU_TO_KELVIN = 315777.09


class units:
    factors = {
        ("angstrom", "angstrom"): 1,
        ("au", "au"): 1,
        ("au2", "au2"): 1,
        ("au", "bohr"): 1,
        ("au2", "bohr2"): 1,
        ("bohr", "au"): 1,
        ("bohr", "bohr"): 1,
        ("bohr2", "bohr2"): 1,
        ("kj_per_mol", "kj_per_mol"): 1,
        ("bohr", "angstrom"): BOHR_TO_ANGSTROM,
        ("bohr2", "angstrom2"): BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM,
        ("au", "angstrom"): BOHR_TO_ANGSTROM,
        ("angstrom", "bohr"): 1 / BOHR_TO_ANGSTROM,
        ("angstrom2", "bohr2"): 1 / (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM),
        ("angstrom", "au"): 1 / BOHR_TO_ANGSTROM,
        ("au", "kj_per_mol"): AU_TO_KJ_PER_MOL,
        ("au", "ev"): AU_TO_EV,
        ("hartree", "kj_per_mol"): AU_TO_KJ_PER_MOL,
        ("hartree", "ev"): AU_TO_EV,
        ("ev", "kj_per_mol"): EV_TO_KJ_PER_MOL,
        ("ev", "au"): 1 / AU_TO_EV,
        ("ev", "hartree"): 1 / AU_TO_EV,
        ("kj_per_mol", "au"): 1 / AU_TO_KJ_PER_MOL,
        ("kj_per_mol", "hartree"): 1 / AU_TO_KJ_PER_MOL,
        ("kj_per_mol", "ev"): 1 / EV_TO_KJ_PER_MOL,
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
        return value * cls._conversion_factor(unit, "bohr")

    @classmethod
    def bohr2(cls, value, unit="au2"):
        unit = cls._s_unit(unit)
        return value * cls._conversion_factor(unit, "bohr2")

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
