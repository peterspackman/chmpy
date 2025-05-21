from chmpy.crystal import Crystal
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import numpy as np

LOG = logging.getLogger("test_describe")


def describe(args, l_max=3):
    name, crystal = args
    name = Path(name).stem
    return (
        name,
        crystal.asymmetric_unit.atomic_numbers,
        crystal.atomic_shape_descriptors(l_max=l_max),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="cif files")
    parser.add_argument("-o", "--output", default="output_{l_max}.npz")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("-l", "--lmax", default=7, help="maximum angular momenta")
    args = parser.parse_args()
    logging.basicConfig(level="INFO")

    crystals = []
    cifs = list(Path(args.directory).glob("*.cif"))
    for path in tqdm(cifs, desc="Loading crystals"):
        try:
            crystals.append((str(path), Crystal.load(str(path))))
        except Exception:
            LOG.error("Error reading %s, skipping", path.name)

    natoms = sum(len(x[1].asymmetric_unit) for x in crystals)
    print("Total atoms in all crystals: ", natoms)
    l_max = args.lmax
    with ProcessPoolExecutor(2) as e:
        descriptors = {}
        futures = [e.submit(describe, crystal, l_max=l_max) for crystal in crystals]
        with tqdm(total=natoms, desc=f"l_max={l_max}", unit="atom") as pbar:
            for f in as_completed(futures):
                name, nums, desc = f.result()
                descriptors[name + "desc"] = desc
                descriptors[name + "element"] = nums
                pbar.update(len(nums))

    output = args.output.format(l_max=l_max)
    print("Saving to ", output)
    np.savez_compressed(output, **descriptors)


if __name__ == "__main__":
    main()
