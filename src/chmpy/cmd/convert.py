import logging
from pathlib import Path
from chmpy import Molecule, Crystal
import sys

LOG = logging.getLogger("chmpy-convert")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-if", "--input-format", default="file_ext")
    parser.add_argument("-of", "--output-format", default="file_ext")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    inpath = Path(args.input)
    outpath = Path(args.output)

    in_kwargs = {}
    if args.input_format != "file_ext":
        in_kwargs["fmt"] = args.input_format
    out_kwargs = {}
    if args.output_format != "file_ext":
        out_kwargs["fmt"] = args.output_format

    x = None
    for cls in Molecule, Crystal:
        try:
            x = cls.load(args.input, **in_kwargs)
            break
        except KeyError as e:
            pass
    else:
        LOG.error("Could not delegate parser for '%s'", args.input)
        sys.exit(1)

    LOG.debug("Loaded %s from %s", x, args.input)

    try:
        x.save(args.output, **out_kwargs)
    except KeyError as e:
        LOG.error("No such writer available (%s) for file '%s'", e, args.output)
        sys.exit(1)

    LOG.debug("Saved %s to %s", x, args.output)


if __name__ == "__main__":
    main()
