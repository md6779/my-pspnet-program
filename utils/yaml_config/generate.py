from argparse import ArgumentParser

from .core import generate_from_yaml_file  # type: ignore

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--cfg", default="./config.yaml", help="Path of config file (.yaml).")
    parser.add_argument("-o", "--out", default="./config.py", help="Path of destination (.py).")

    args = parser.parse_args()

    generate_from_yaml_file(args.cfg, args.out)
