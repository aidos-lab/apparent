"""Ollivier-Ricci based analysis of Physician Networks."""

import argparse
import os
import sys

import config
import pandas as pd

import pickle
import swifter
from curvature import (
    forman_curvature,
    ollivier_ricci_curvature,
    resistance_curvature,
)


def format_column_labels(args):
    """
    Formats the column labels based on the given arguments.

    Args:
        args (Namespace): The namespace object containing the arguments.

    Returns:
        str: The formatted column label.

    Raises:
        ValueError: If the curvature type is unsupported.
    """
    label = "Unlabeled"
    if args.curvature == "OR":
        if args.prob_fn is not None:
            label = f"{args.curvature}_{args.prob_fn}"
        else:  # Alpha is only for default probability measure
            label = f"{args.curvature}_{args.alpha}"
    elif args.curvature == "Forman" or args.curvature == "Resistance":
        label = f"{args.curvature}"
    else:
        raise ValueError(f"Unsupported curvature type: {args.curvature}")
    return label


def calculate_curvature(G, curvature_type, alpha=0, prob_fn=None):
    """
    Calculate the curvature of a graph based on the specified curvature type.

    Parameters:
        row (Graph): The graph for which to calculate the curvature.
        curvature_type (str): The type of curvature to calculate. Supported types are "OR" (Ollivier-Ricci) and "Forman".
        alpha (float, optional): The alpha parameter for Ollivier-Ricci curvature. Default is 0.
        prob_fn (function, optional): The probability function for Ollivier-Ricci curvature. Default is None.

    Returns:
        float: The calculated curvature value.

    Raises:
        ValueError: If an unsupported curvature type is specified.

    """
    if curvature_type == "OR":
        return ollivier_ricci_curvature(G=G, alpha=alpha, prob_fn=prob_fn)
    elif curvature_type == "Forman":
        return forman_curvature(G=G)
    elif curvature_type == "Resistance":
        return resistance_curvature(G=G)
    else:
        raise ValueError(f"Unsupported curvature type: {curvature_type}")


def add_curvature_feature(feature_dict, args):
    """
    Function to add a curvature column to a DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame to which the curvature column will be added.
    - args (Namespace): Namespace containing arguments including:
        - data (str): Name of the column where the data is stored.
        - sample_size (int): Number of samples to compute curvature for.
        - curvature (str): Type of curvature calculation.
        - alpha (float): Alpha value for curvature calculation.
        - prob_fn (callable): Probability function for curvature calculation.
        - save (bool): Whether to save the DataFrame after adding the column.

    Returns:
    - pd.DataFrame: DataFrame with the curvature column added.
    """

    assert (
        "graph" in feature_dict.keys()
    ), "Graph not found in feature dictionary. First build by running `build_networks.py`."

    G = feature_dict["graph"]
    label = format_column_labels(args)

    if label not in feature_dict.keys():
        feature_dict[label] = calculate_curvature(
            G=G,
            curvature_type=args.curvature,
            alpha=args.alpha,
            prob_fn=args.prob_fn,
        )
    else:
        print(f"{label} feature is already computed!.")
    return feature_dict


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

    # Read data from pickle file
    files = config.OUTPUT_PATH + "graphs/"

    assert os.path.exists(files), f"Path not found: {files}"

    for file in os.listdir(files):
        file = os.path.join(files, file)
        print("Computing Curvature for file: ", file)
        if file.endswith(".pkl"):
            with open(file, "rb") as r:
                feature_dict = pickle.load(r)
            result = add_curvature_feature(feature_dict, args)
            with open(file, "wb") as w:
                pickle.dump(result, w)
                continue

    print(f"New Column Added: {label}")
