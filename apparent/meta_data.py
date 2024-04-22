import argparse
import os
import sys
import pandas as pd
import config
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--graphs_dir",
        type=str,
        default=config.OUTPUT_PATH + "graphs/",
        help="Dataset.",
    )
    parser.add_argument(
        "--network_data",
        type=str,
        default=config.OUTPUT_PATH + "networks_data.csv",
        help="Dataset.",
    )

    parser.add_argument(
        "--new_data",
        type=str,
        default=config.DATA_PATH + "hsa_mortality_dead6599ffs_2017.xlsx",
        help="CSV Data File.",
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2017,
        help="Year!",
    )

    parser.add_argument(
        "--new_index",
        type=str,
        default="HSA #",
        help=" Excel Column on which to join the dataframes.",
    )
    parser.add_argument(
        "--excel_sheet",
        type=int,
        default=1,
        help="Which Excel Sheet to query on.",
    )

    parser.add_argument(
        "--join",
        type=str,
        default="hsanum",
        help="Column on which to join the dataframes.",
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    assert os.path.exists(
        args.network_data
    ), f"Path not found: {args.network_data}"
    assert os.path.exists(args.new_data), f"Path not found: {args.new_data}"

    # Read data from pickle file
    print("Graphs Directory: ", args.graphs_dir)
    assert os.path.exists(args.graphs_dir), f"Path not found: {args.graphs_dir}"

    files = os.listdir(args.graphs_dir)

    # Sort the files
    sorted(files)
    # Read in the network data and new data

    all_network_data = pd.read_csv(args.network_data, index_col=0)
    network_data = all_network_data[all_network_data["year"] == args.year]
    new_data = pd.read_excel(args.new_data, sheet_name=args.excel_sheet)

    joined_data = network_data.join(
        new_data.set_index(args.new_index), on=args.join
    )

    for i, row in joined_data.iterrows():
        data = {col: row[col] for col in joined_data.columns}

        out_file = os.path.join(args.graphs_dir, f"graph_{i}.pkl")
        assert os.path.exists(
            out_file
        ), "Please compute graphs using `build_networks.py`"

        assert data["year"] == args.year, "Year mismatch!"

        with open(out_file, "rb") as f:
            saved_network = pickle.load(f)
        data = {**saved_network, **data}
        with open(out_file, "wb") as f:
            pickle.dump(data, f)
print("Finished pickling individual graphs!")
