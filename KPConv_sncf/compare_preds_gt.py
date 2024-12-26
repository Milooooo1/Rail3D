from numpy.lib import recfunctions as rfn
from utils.ply import read_ply, write_ply
import numpy as np
import pathlib
import argparse

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input", type=str, required=True, help="Path to the input .ply file")
    argparser.add_argument("-o", "--output", type=str, help="Path to the output .ply file (optional)")

    args = argparser.parse_args()

    # Parse arguments
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output) if args.output else input_path

    if not input_path.is_file() or input_path.suffix != ".ply":
        print("\033[91mERROR\033[0m: The input file is not a valid .ply file.")
        exit()

    # Read data and compute errors
    data = read_ply(str(input_path))

    if 'preds' not in data.dtype.names or 'class' not in data.dtype.names:
        print("\033[91mERROR\033[0m: The input file does not contain 'preds' or 'class' columns.")
        exit()

    preds = data['preds']
    ground_truth = data['class']

    errors = np.where(preds != ground_truth, 1, 0)
    data = rfn.append_fields(data, 'errors', errors, usemask=False)

    # Export
    field_names = list(data.dtype.names)
    field_list = [data[name] for name in field_names]
    write_ply(str(output_path), field_list, field_names)

if __name__ == "__main__":
    main()