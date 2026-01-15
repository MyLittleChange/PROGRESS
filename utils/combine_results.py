import json
import argparse
import os
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
torch.backends.cudnn.benchmark = False     # ensures deterministic behavior
# Optional: Set environment variable for any libraries that check it
import os
os.environ['PYTHONHASHSEED'] = str(seed)


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
def combine_results(input_path, is_internVL=False, num_gpus=4):
    if is_internVL:
        file_paths = []
        for i in range(num_gpus):
            file_paths.append(os.path.join(input_path, f"{num_gpus}_{i}_internVL.json"))
        print(file_paths)
    else:
        file_paths = [
            os.path.join(input_path, "4_0.json"),
            os.path.join(input_path, "4_1.json"),
            os.path.join(input_path, "4_2.json"),
            os.path.join(input_path, "4_3.json"),
        ]
    combined_data = {
        "cluster_accuracies": {},
        "total_predictions": 0,
        "overall_accuracy": 0,
    }
    for path in file_paths:
        json_data = read_json_file(path)
        print(path)
        for cluster_id, cluster_data in json_data["cluster_accuracies"].items():
            if cluster_id not in combined_data["cluster_accuracies"]:
                combined_data["cluster_accuracies"][cluster_id] = cluster_data
            else:
                combined_data["cluster_accuracies"][cluster_id][
                    "correct"
                ] += cluster_data["correct"]
                combined_data["cluster_accuracies"][cluster_id][
                    "total"
                ] += cluster_data["total"]
    for cluster_id, cluster_data in combined_data["cluster_accuracies"].items():
        combined_data["cluster_accuracies"][cluster_id]["accuracy"] = (
            combined_data["cluster_accuracies"][cluster_id]["correct"]
            / combined_data["cluster_accuracies"][cluster_id]["total"]
        )
    combined_data["cluster_accuracies"] = {
        k: v
        for k, v in sorted(
            combined_data["cluster_accuracies"].items(),
            key=lambda item: item[1]["accuracy"],
            reverse=True,
        )
    }
    combined_data["overall_accuracy"] = sum(
        cluster_data["correct"]
        for cluster_data in combined_data["cluster_accuracies"].values()
    ) / sum(
        cluster_data["total"]
        for cluster_data in combined_data["cluster_accuracies"].values()
    )
    if is_internVL:
        with open(os.path.join(input_path, "combined_results_internVL.json"), "w") as f:
            json.dump(combined_data, f, indent=4)
    else:
        with open(os.path.join(input_path, "combined_results.json"), "w") as f:
            json.dump(combined_data, f, indent=4)


def min_max_normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    return [(x - min_score) / (max_score - min_score) for x in scores]


def z_score_normalize(scores):
    mean = sum(scores) / len(scores)
    std = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5
    return [(x - mean) / std for x in scores]


def sum_normalize(scores):
    total = sum(scores)
    return [score / total for score in scores]


def combine_results_confidence(input_path, num_gpus):
    file_paths = []
    # detect the number of GPUs used
    for i in range(num_gpus):
        file_paths.append(os.path.join(input_path, f"confidence_{num_gpus}_{i}.json"))
    total_confidence_scores = []
    total_cluster_ids = []
    for path in file_paths:
        json_data = read_json_file(path)
        total_confidence_scores.extend(json_data["confidence_scores"])
        total_cluster_ids.extend(json_data["cluster_ids"])
    # Apply different normalizations
    cluster_losses = {}
    for loss, cluster_id in zip(total_confidence_scores, total_cluster_ids):
        if cluster_id not in cluster_losses:
            cluster_losses[cluster_id] = []
        cluster_losses[cluster_id].append(loss)

    # Calculate average loss per cluster
    cluster_avg_losses = {
        cluster_id: sum(losses) / len(losses)
        for cluster_id, losses in cluster_losses.items()
    }
    # total_loss = sum(cluster_avg_losses.values())
    cluster_accuracy_scores = {
        cluster_id: {
            "accuracy": loss,
            "total": len(cluster_losses[cluster_id]),
        }
        for cluster_id, loss in cluster_avg_losses.items()
    }
    cluster_accuracy_scores = {
        k: v
        for k, v in sorted(
            cluster_accuracy_scores.items(),
            key=lambda item: item[1]["accuracy"],
            reverse=True,
        )
    }

    res = {
        "cluster_accuracies": cluster_accuracy_scores,
        "cluster_avg_losses": cluster_avg_losses,
        "cluster_ids": total_cluster_ids,
        "total_confidence_scores": total_confidence_scores,
    }
    res = json.dumps(res, indent=4)
    with open(os.path.join(input_path, "combined_confidence.json"), "w") as f:
        f.write(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="/home/mila/q/qian.yang/scratch/visual_imagination/accuracy_based_selection_COINCIDE_bert_10k_clusters_total_200K_warmup_133K",
    )
    parser.add_argument(
        "--is_internVL",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--is_confidence",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=4,
    )
    args = parser.parse_args()
    if args.is_confidence:
        combine_results_confidence(args.input_path, args.num_gpus)
    else:
        combine_results(args.input_path, args.is_internVL, args.num_gpus)
