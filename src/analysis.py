"""Ollivier-Ricci based analysis of Physician Networks."""

import argparse
import os
import sys
import numpy as np
import pickle
import pandas as pd
import config

from nammu.curvature import ollivier_ricci_curvature


# Eventual Goal is to read from existing csv file, and add a column of curvature values for each alpha value
# For now will just save a dictionary of key = ? : G,{curvature_values} for single alpha value
# if files are too big we can split by year

cwd = os.path.dirname(__file__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Select location of input file.",
    )

    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        help="Alpha parameter for OR Curvature",
    )

    parser.add_argument(
        "--row_sample",
        default=5,
        type=int,
        help="Number of columns to sample from full datafile.",
    )

    parser.add_argument(
        "-v",
        "--Verbose",
        default=True,
        action="store_true",
        help="If set, will print messages detailing computation and output.",
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    assert os.path.isfile(args.data), "Invalid Input Data"
    # Load Graphs
    with open(args.data, "rb") as f:
        data = pickle.load(f)
        # For Local Testing on Data Subsample
        if args.row_sample:
            data = data.sample(n=args.row_sample, axis="rows")

    results = {}
    # Use index in original csv as identifier
    for i, G in zip(data.index, data["G"]):
        curvature = ollivier_ricci_curvature(G, alpha=args.alpha)
        results[i] = (G, curvature)

    output_dir = os.path.join(cwd, config.OUTPUT_PATH)
    output_file = f"or_curvature_alpha{args.alpha}.pkl"

    if os.path.isdir(output_dir):
        output_file = os.path.join(output_dir, output_file)
    else:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_file)

    with open(output_file, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("\n")

    out_dir_message = "/".join(output_file.split("/")[-2:])

    if args.Verbose:
        print("\n")
        print(
            "-------------------------------------------------------------------------------- \n\n"
        )
        print(
            f"Successfully written curvature analysis output to {out_dir_message}"
        )

        print(
            "\n\n -------------------------------------------------------------------------------- "
        )
