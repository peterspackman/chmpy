"""Write src/chmpy/crystal/hall_symbols.json: a Hall symbol for every setting.

Each setting in sgdata.json is matched by its set of symmetry operations to a
gemmi space group table entry. gemmi is needed only to run this script.

    python scripts/generate_hall_symbols.py
"""

import json
from pathlib import Path

import gemmi

from chmpy.crystal.symmetry_operation import SymmetryOperation

CRYSTAL = Path(__file__).parent.parent / "src" / "chmpy" / "crystal"


def main():
    hall_by_ops = {}
    for entry in gemmi.spacegroup_table():
        codes = frozenset(
            SymmetryOperation.from_string_code(op.triplet()).integer_code
            for op in entry.operations()
        )
        hall_by_ops.setdefault(codes, entry.hall)

    settings = json.loads((CRYSTAL / "sgdata.json").read_text())
    result = {}
    for number, entries in settings.items():
        for setting in entries:
            key = f"{number}:{setting[6]}"
            result[key] = hall_by_ops[frozenset(setting[8])]
    (CRYSTAL / "hall_symbols.json").write_text(json.dumps(result, indent=0) + "\n")
    print(f"Wrote {len(result)} Hall symbols")


if __name__ == "__main__":
    main()
