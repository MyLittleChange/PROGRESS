import json
import argparse
import random
import torch

seed = 42
# PyTorch random number generator
torch.manual_seed(seed)
# CUDA random number generator
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multi-GPU
# Python random number generator
import random

random.seed(seed)
# NumPy random number generator
import numpy as np

np.random.seed(seed)
# Additional settings for reproducibility
torch.backends.cudnn.deterministic = True  # ensures deterministic behavior
torch.backends.cudnn.benchmark = False  # ensures deterministic behavior
# Optional: Set environment variable for any libraries that check it
import os

os.environ["PYTHONHASHSEED"] = str(seed)


def random_select_samples(file_path, output_path, num_samples_per_cluster=50):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cluster_ids = [item["cluster_id"] for item in data]
    print("original number of samples", len(data))
    # get the number of samples per cluster
    index_per_cluster = {}
    for i, cluster_id in enumerate(cluster_ids):
        if cluster_id not in index_per_cluster:
            index_per_cluster[cluster_id] = []
        index_per_cluster[cluster_id].append(i)
    # randomly select
    selected_data_per_cluster = []
    for cluster_id in index_per_cluster:
        num_samples = len(index_per_cluster[cluster_id])
        # at least 50 samples
        # if num_samples > 100:
        #     num_samples_to_select = max(100, num_samples // 4)
        # else:
        #     num_samples_to_select = num_samples

        num_samples_to_select = min(num_samples, num_samples_per_cluster)
        selected_indices = random.sample(
            index_per_cluster[cluster_id], num_samples_to_select
        )
        selected_data = [data[i] for i in selected_indices]
        selected_data_per_cluster.extend(selected_data)
    random.shuffle(selected_data_per_cluster)
    print("length of randomly selected data", len(selected_data_per_cluster))
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(selected_data_per_cluster, outfile, ensure_ascii=False, indent=4)


def concatenate_json_files(file1_path, file2_path, output_path):
    # Read first file
    with open(file1_path, "r", encoding="utf-8") as f1:
        data1 = json.load(f1)
    print(len(data1))
    # Read second file
    with open(file2_path, "r", encoding="utf-8") as f2:
        data2 = json.load(f2)
    print(len(data2))
    # Concatenate the lists
    combined_data = data1 + data2

    # Write to output file
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(combined_data, outfile, ensure_ascii=False, indent=4)

    print(f"Successfully combined files. Total entries: {len(combined_data)}")


def remove_samples_from_json(file1_path, file2_path, output_path):
    with open(file1_path, "r", encoding="utf-8") as f1:
        data1 = json.load(f1)
    with open(file2_path, "r", encoding="utf-8") as f2:
        data2 = json.load(f2)

    # remove samples from data1 that are present in data2
    print("Total Samples in file1", len(data1))
    print("Samples to remove", len(data2))
    data = [item for item in data1 if item not in data2]
    print("Total Samples in output file", len(data))
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


## randomly select desired number of samples from a json file and save to a new file
def select_random_samples(
    file_path, num_samples, output, exclude_clusters=None, uniform_select=False
):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("length of data", len(data))
    # exclude clusters
    if exclude_clusters is not None:

        print("exclude_clusters", exclude_clusters)
        data = [item for item in data if item["cluster_id"] not in exclude_clusters]
        print("length of data after excluding clusters", len(data))

    if uniform_select:
        cluster_ids = set([item["cluster_id"] for item in data])
        num_clusters = len(cluster_ids)
        print("num_clusters", num_clusters)
        # get the number of samples per cluster
        samples_per_cluster = num_samples // num_clusters
        print("samples_per_cluster", samples_per_cluster)
        # uniformly select samples from each cluster
        # import pdb; pdb.set_trace()
        selected_data = []
        for cluster_id in cluster_ids:
            cluster_data = [item for item in data if item["cluster_id"] == cluster_id]
            # import pdb; pdb.set_trace()
            selected_data.extend(random.sample(cluster_data, samples_per_cluster))
        data = selected_data
        print("length of data after uniformly selecting", len(data))

    random.shuffle(data)

    selected_data = data[:num_samples]
    print("length of selected data", len(selected_data))
    with open(output, "w", encoding="utf-8") as outfile:
        json.dump(selected_data, outfile, ensure_ascii=False, indent=4)
    return selected_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate two JSON files into one.")
    parser.add_argument("--file1", help="Path to the first JSON file")
    parser.add_argument("--file2", help="Path to the second JSON file")
    parser.add_argument("--output", help="Path to the output JSON file")
    parser.add_argument(
        "--num_samples", type=int, help="Number of samples to randomly select"
    )
    parser.add_argument(
        "--randomly_select",
        action="store_true",
        help="Randomly select samples from the first file",
    )
    parser.add_argument(
        "--concatenate", action="store_true", help="Concatenate the two files"
    )
    parser.add_argument(
        "--remove_samples",
        action="store_true",
        help="Remove samples from the first file that are present in the second file",
    )
    parser.add_argument(
        "--exclude_clusters",
        type=lambda x: [int(i) for i in x.split(",")],
        default=None,  # Convert comma-separated string to list of integers
        help='Comma-separated list of cluster numbers to exclude (e.g., "1,2,3")',
    )
    parser.add_argument(
        "--uniform_select",
        action="store_true",
        help="Uniformly select samples from the clusters",
    )
    parser.add_argument(
        "--random_select_samples",
        action="store_true",
        help="Randomly select samples from the first file",
    )

    parser.add_argument(
        "--samples_per_cluster",
        type=int,
        default=50,
        help="Number of samples to select per cluster using REPR",
    )
    parser.add_argument(
        "--dino_feature_path",
        type=str,
        default="/home/mila/q/qian.yang/scratch/llava-v1.5-7b/instruct_tuning_data/dino_v2_l_feats.npy",
        help="Path to the DINO features",
    )
    parser.add_argument(
        "--bert_feature_path",
        type=str,
        default="/home/mila/q/qian.yang/scratch/llava-v1.5-7b/instruct_tuning_data/q_features.npy",
        help="Path to the BERT features",
    )
    parser.add_argument(
        "--kmeans_ratio",
        type=int,
        default=4,
        help="Ratio of kmeans clusters to final samples",
    )
    parser.add_argument(
        "--combine_visual_imagination_data",
        action="store_true",
        help="Combine visual imagination data with the first file",
    )

    args = parser.parse_args()

    if args.randomly_select:
        select_random_samples(
            args.file1,
            args.num_samples,
            args.output,
            args.exclude_clusters,
            args.uniform_select,
        )
    if args.concatenate:
        concatenate_json_files(args.file1, args.file2, args.output)
    if args.remove_samples:
        remove_samples_from_json(args.file1, args.file2, args.output)
    if args.random_select_samples:
        random_select_samples(args.file1, args.output, args.samples_per_cluster)