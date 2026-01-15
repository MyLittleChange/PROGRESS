import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from PIL import Image
import random


def read_json_file(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON")
        return None


# Example usage
def combine_results(
    nearest_centroid_path,
    chosen_indices_path,
    dist_to_cent_path,
    llava_data_path,
    output_path,
    keep_chosen_only=False,
):
    print("Loading data from: ", llava_data_path)
    llava_data = read_json_file(llava_data_path)
    print("Size of all data: ", len(llava_data))

    # Handle cases based on what paths are provided

    if nearest_centroid_path is not None:
        # Full clustering workflow
        nearest_centroid = np.load(nearest_centroid_path)
        dist_to_cent = np.load(dist_to_cent_path) if dist_to_cent_path else None

        assert len(nearest_centroid) == len(
            llava_data
        ), "The length of nearest_centroid and llava_data must be the same"

        for i in range(len(nearest_centroid)):
            llava_data[i]["cluster_id"] = int(nearest_centroid[i])
            if dist_to_cent is not None:
                llava_data[i]["dist_to_cent"] = float(dist_to_cent[i])
            llava_data[i]["index"] = i
        if keep_chosen_only:
            assert chosen_indices_path is not None
            chosen_accuracy_indices = np.load(chosen_indices_path)

            llava_data = [llava_data[i] for i in chosen_accuracy_indices]
    else:
        if keep_chosen_only:
            assert chosen_indices_path is not None
            chosen_accuracy_indices = np.load(chosen_indices_path)
            print("Filtering to chosen indices only (without cluster information).")
            llava_data = [llava_data[i] for i in chosen_accuracy_indices]
        else:
            print("Output will contain original data without cluster information.")

    llava_data = json.dumps(llava_data, indent=4)
    print("Saving to: ", output_path)
    with open(output_path, "w") as f:
        f.write(llava_data)
    # read the output file and print the first 10 samples
    llava_data = read_json_file(output_path)
    print("Final llava_data size: ", len(llava_data))


# import pdb; pdb.set_trace()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist_to_cent_path",
        type=str,
        default=None,
        help="Path to distance to centroid file (optional)",
    )
    parser.add_argument(
        "--nearest_centroid_path",
        type=str,
        default=None,
        help="Path to nearest centroid file (optional)",
    )
    parser.add_argument(
        "--chosen_indices_path",
        type=str,
        default=None,
        help="Path to chosen indices file (optional)",
    )
    parser.add_argument(
        "--llava_data_path",
        type=str,
        default="llava_v1_5_mix665k.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="llava_v1_5_mix665k_dino-bert_10K_Leftover_605K.json",
    )
    parser.add_argument(
        "--keep_chosen_only",
        action="store_true",
        help="If set, only keep the chosen indices in the output file",
    )

    args = parser.parse_args()
    combine_results(
        args.nearest_centroid_path,
        args.chosen_indices_path,
        args.dist_to_cent_path,
        args.llava_data_path,
        args.output_path,
        args.keep_chosen_only,
    )
