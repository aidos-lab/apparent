"""Ollivier-Ricci based analysis of Physician Networks."""

import argparse
import os
import sys

import config
import pandas as pd
import swifter
from curvature import forman_curvature, ollivier_ricci_curvature


def format_column_labels(args):
    label = "Unlabeled"
    if args.curvature == "OR":
        if args.prob_fn is not None:
            label = f"{args.curvature}_{args.prob_fn}"
        else:  # Alpha is only for defualt probability measure
            label = f"{args.curvature}_{args.alpha}"

    elif args.curvature == "Forman":
        label = f"{args.curvature}"
    else:
        raise ValueError(f"Unsupported curvature type: {args.curvature}")
    return label


def calculate_curvature(row, curvature_type, alpha=0, prob_fn=None):
    if curvature_type == "OR":
        return ollivier_ricci_curvature(G=row, alpha=alpha, prob_fn=prob_fn)
    elif curvature_type == "Forman":
        return forman_curvature(G=row)  # Adjust as needed for Forman curvature
    else:
        raise ValueError(f"Unsupported curvature type: {curvature_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="nx_networks_undirected_local_hsa.pkl",
        help="Dataset.",
    )

    parser.add_argument(
        "-c",
        "--curvature",
        type=str,
        default="OR",
        help="Type of curvature to compute.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of rows to process.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        help="Alpha value for ollivier-ricci computation.",
    )

    parser.add_argument(
        "--prob_fn",
        type=str,
        default=None,
        help="Probability function for ollivier-ricci computation.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save results by using `-s`.",
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    n_cpus = os.cpu_count() - 4
    print(f"Uing {n_cpus} cpus for computation.")

    df = pd.read_pickle(config.DATA_PATH + args.data)
    label = format_column_labels(args)

    if label not in df.columns:
        df[label] = [None] * df.shape[0]
    else:
        first_empty_index = df[label].isna().idxmax()

    partial = df.iloc[first_empty_index : first_empty_index + args.sample_size]

    print("Calculating Curvature...")
    partial = partial.assign(
        **{
            label: partial.G.swifter.apply(
                lambda row: calculate_curvature(
                    row, args.curvature, alpha=args.alpha, prob_fn=args.prob_fn
                )
            )
        },
    )

    # replace
    df.iloc[first_empty_index : first_empty_index + args.sample_size] = partial

    print(df[label])
    print(df.head())
    if args.save:
        print("Writing Pickle...")

        df.to_pickle(config.DATA_PATH + args.data)

        print("Finished pickling")
        print("----------------------------------")
        print(f"New Column Added: {label}")
        print(f"Dataframe shape: {df.shape}")
