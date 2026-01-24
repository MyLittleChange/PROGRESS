import json
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import numpy as np
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acc_file",
        type=str,
        default="/home/mila/q/qian.yang/scratch/visual_imagination/llava_lora/llava_lora_warmup_60K_total_133k_10K_clusters_AllSelection_Fastest_Prev/selection_step_469/combined_results_internVL.json",
    )
    parser.add_argument(
        "--visualize_path",
        type=str,
        default="/home/mila/q/qian.yang/scratch/visual_imagination/accuracy_based_selection_COINCIDE_bert_10k_clusters_total_200K_warmup_133K/warm_up/visualize/",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/network/scratch/q/qian.yang/llava-v1.5-7b/instruct_tuning_data/data",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/mila/q/qian.yang/scratch/visual_imagination/llava_lora/tmp.json",
    )
    parser.add_argument(
        "--output_selected_cluster_accuracies",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_visual_imagination_data",
        type=str,
        default=None,
    )
    parser.add_argument("--do_sort", action="store_true")

    parser.add_argument("--selection_mid", action="store_true")
    parser.add_argument("--selection_hardest", action="store_true")
    parser.add_argument("--selection_easiest", action="store_true")
    parser.add_argument("--selection_num", type=int, default=7500)


    parser.add_argument(
        "--selection_file",
        type=str,
        default="/home/mila/q/qian.yang/scratch/llava-v1.5-7b/instruct_tuning_data/All_LBA_1K_7K_Prev/llava_v1_5_mix665k_dino-bert_1K_Top_60K.json",
    )
    parser.add_argument(
        "--previous_acc_file",
        type=str,
        default="/home/mila/q/qian.yang/scratch/visual_imagination/llava_lora/llava_lora_warmup_60K_total_133k_1K_clusters_AllSelection_Fastest_Prev/selection_step_391/combined_results_internVL.json",
    )
    parser.add_argument("--compare_with_previous", action="store_true")
    parser.add_argument("--easy_mid", action="store_true")
    parser.add_argument("--base_path", type=str, default="")
    parser.add_argument(
        "--output_average_acc_change",
        type=str,
        default="/home/mila/q/qian.yang/scratch/visual_imagination/llava_lora/llava_lora_warmup_60K_total_133k_1K_clusters_AllSelection_Fastest_Prev/selection_step_469/acc_change_overall.json",
    )
    parser.add_argument("--selection_relative_change", action="store_true")
    parser.add_argument("--softmax_selection", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--step_list_file", type=str, default=None)
    parser.add_argument("--none_LBA", type=float, default=0.05)
    parser.add_argument("--use_loss", action="store_true")
    parser.add_argument("--use_current_acc", action="store_true")
    return parser.parse_args()


def main(args):

    print("Reading accuracy file", args.acc_file)
    with open(args.acc_file, "r") as f:
        acc_data = json.load(f)
    if args.compare_with_previous:
        with open(args.previous_acc_file, "r") as f:
            previous_acc_data = json.load(f)
        with open(args.acc_file, "r") as f:
            current_acc_data = json.load(f)
        if args.output_selected_cluster_accuracies is not None:
            with open(args.output_selected_cluster_accuracies, "r") as f:
                selected_cluster_accuracies = json.load(f)
            selected_cluster_ids = [
                cluster_id for cluster_id in selected_cluster_accuracies.keys()
            ]
        else:
            selected_cluster_ids = [
                cluster_id
                for cluster_id in current_acc_data["cluster_accuracies"].keys()
            ]
        previous_acc = {
            cluster_id: (
                previous_acc_data["cluster_accuracies"][cluster_id]["accuracy"]
                if cluster_id in previous_acc_data["cluster_accuracies"]
                else 0
            )
            for cluster_id in selected_cluster_ids
        }
        current_acc = {}
        for cluster_id in selected_cluster_ids:
            try:
                current_acc[cluster_id] = current_acc_data["cluster_accuracies"][
                    cluster_id
                ]["accuracy"]
            except (KeyError, TypeError):
                current_acc[cluster_id] = 0
        print(previous_acc)
        print("-" * 100)
        print(current_acc)
        average_acc_change = {
            cluster_id: current_acc[cluster_id] - previous_acc[cluster_id]
            for cluster_id in selected_cluster_ids
        }
        # sort by average_acc_change and keep it a dictionary
        average_acc_change = dict(
            sorted(average_acc_change.items(), key=lambda x: x[1], reverse=True)
        )
        total_num_per_cluster = {
            cluster_id: current_acc_data["cluster_accuracies"][cluster_id]["total"]
            for cluster_id in selected_cluster_ids
        }
        for cluster_id in selected_cluster_ids:
            if previous_acc[cluster_id] == 0:
                previous_acc[cluster_id] = 1 / total_num_per_cluster[cluster_id]
        if args.use_loss:
            if args.use_current_acc:
                relative_acc_change = {
                    cluster_id: (
                        (current_acc[cluster_id] - previous_acc[cluster_id])
                        / current_acc[cluster_id]
                    )
                    for cluster_id in selected_cluster_ids
                }
            else:
                relative_acc_change = {
                    cluster_id: (
                        (current_acc[cluster_id] - previous_acc[cluster_id])
                        / previous_acc[cluster_id]
                    )
                    for cluster_id in selected_cluster_ids
                }
        else:
            relative_acc_change = {
                cluster_id: (
                    (current_acc[cluster_id] - previous_acc[cluster_id])
                    / previous_acc[cluster_id]
                )
                for cluster_id in selected_cluster_ids
            }
        # sort by relative_acc_change and keep it a dictionary
        relative_acc_change = dict(
            sorted(relative_acc_change.items(), key=lambda x: x[1], reverse=True)
        )
        acc_change = {
            "average_acc_change": average_acc_change,
            "relative_acc_change": relative_acc_change,
            # "relative_acc_change_None_LBA": relative_acc_change_None_LBA,
        }
        # print net change
        net_change = sum(average_acc_change.values())
        print(f"Net change: {net_change:.4f}")
        with open(args.output_average_acc_change, "w") as f:
            json.dump(acc_change, f, indent=4)

    if args.visualize:
        # sort clusters as per accuracy
        with open(args.output_average_acc_change, "r") as f:
            acc_change = json.load(f)
        relative_acc_change = acc_change["relative_acc_change"]

        # Filter out negative changes and get total positive change
        positive_changes = {k: v for k, v in relative_acc_change.items() if v > 0}
        if args.visualize:
            cluster_list = list(positive_changes.keys())
            print(f"Visualizing clusters: {cluster_list}")
        else:
            sorted_clusters = sorted(
                acc_data["cluster_accuracies"].items(),
                key=lambda x: x[1]["accuracy"],
                reverse=True,
            )
            # select top 5, middle 5, bottom
            print(len(sorted_clusters))
            top_5 = sorted_clusters[:5]
            print(top_5)
            mid_idx = len(sorted_clusters) // 2
            middle_5 = sorted_clusters[mid_idx - 2 : mid_idx + 3]
            print(middle_5)
            bottom_5 = sorted_clusters[-5:]
            print(bottom_5)
            cluster_list = top_5 + middle_5 + bottom_5
            print(cluster_list)
        # visualize 5 samples from each cluster
        if not os.path.exists(args.visualize_path):
            os.makedirs(args.visualize_path)
        cluster_list = cluster_list[:10]
        for i, (cluster_id) in enumerate(cluster_list):
            # get data from selection file
            data = json.load(open(args.selection_file, "r"))
            cluster_data = [
                sample for sample in data if sample["cluster_id"] == int(cluster_id)
            ]

            for sample in cluster_data[:10]:
                try:
                    if "image" not in sample.keys():
                        image = Image.new("RGB", (256, 256), (255, 255, 255))
                    else:
                        image = Image.open(
                            os.path.join(args.image_dir, sample["image"])
                        )
                except Exception as e:
                    print(f"Error loading image: {e}")
                    import pdb

                    pdb.set_trace()
                    continue
                conversation = sample["conversations"]
                questions = []
                for conv in conversation:
                    if conv["from"] == "human":
                        questions.append(conv["value"])
                question = "\n".join(questions)
                # plot image and question together and save it
                # import pdb; pdb.set_trace()
                plt.figure(figsize=(10, 12))
                plt.subplot(2, 1, 1)
                plt.imshow(image)
                plt.axis("off")
                # text start from the top left
                plt.subplot(2, 1, 2)
                # large font size
                plt.text(0, 0.5, question, wrap=True, fontsize=22)
                plt.axis("off")
                if not os.path.exists(args.visualize_path + f"cluster_{i}"):
                    os.makedirs(args.visualize_path + f"cluster_{i}")

                sample_id = str(sample["id"]).replace("/", "_")
                try:
                    plt.savefig(
                        os.path.join(
                            args.visualize_path,
                            f"cluster_{i}",
                            f"accuracy_{cluster_id}" + f"_sample_{sample_id}.png",
                        )
                    )
                except Exception as e:
                    print(f"Error saving image: {e}")
                plt.close()
                # import pdb; pdb.set_trace()

                print("-" * 100)

    if args.selection_mid:
        print("Selection mid")
        # Sort clusters by accuracy
        cluster_accuracies = acc_data["cluster_accuracies"]
        sorted_clusters = sorted(
            cluster_accuracies.items(), key=lambda x: x[1]["accuracy"]
        )
        data = json.load(open(args.selection_file, "r"))
        cluster_accuracies = acc_data["cluster_accuracies"]
        sample_num_per_cluster = {}
        for sample in data:
            if sample["cluster_id"] not in sample_num_per_cluster:
                sample_num_per_cluster[sample["cluster_id"]] = 1
            else:
                sample_num_per_cluster[sample["cluster_id"]] += 1
        print(sample_num_per_cluster)
        # remove clusters from sorted_clusters with 0 samples

        sorted_clusters = [
            cluster
            for cluster in sorted_clusters
            if int(cluster[0]) in sample_num_per_cluster
        ]

        if args.easy_mid:
            mid = len(sorted_clusters) // 2
            mid_idx = (mid + len(sorted_clusters)) // 2
        else:
            mid_idx = len(sorted_clusters) // 2

        # Initialize variables for selecting clusters
        selected_cluster_ids = []
        total_samples = 0
        left = mid_idx - 1
        right = mid_idx

        print("Lowest cluster accuracy is: ", sorted_clusters[0][1]["accuracy"])
        print("Middle cluster accuracy is: ", sorted_clusters[mid_idx][1]["accuracy"])
        print("Highest cluster accuracy is: ", sorted_clusters[-1][1]["accuracy"])
        # Select clusters alternating between left and right until we reach target sample count
        expand_right = True  # Flag to alternate between left and right
        while total_samples < args.selection_num and (
            left >= 0 or right < len(sorted_clusters)
        ):
            if expand_right and right < len(sorted_clusters):
                cluster_id = int(sorted_clusters[right][0])
                right += 1
                if cluster_id not in sample_num_per_cluster:
                    continue
                selected_cluster_ids.append(cluster_id)
                total_samples += sample_num_per_cluster[cluster_id]

            elif not expand_right and left >= 0:
                cluster_id = int(sorted_clusters[left][0])
                left -= 1
                if cluster_id not in sample_num_per_cluster:
                    continue
                selected_cluster_ids.append(cluster_id)
                total_samples += sample_num_per_cluster[cluster_id]
            expand_right = not expand_right  # Switch direction for next iteration
            print(total_samples)

        # get the accuracy of selected_cluster_ids
        # selected_cluster_ids = ['50', '49', '48', '47', '46', '45', '44', '43', '42', '41']
        selected_cluster_accuracies = {
            cluster_id: cluster_accuracies[str(cluster_id)]
            for cluster_id in selected_cluster_ids
        }
        # sort cluster ids by accuracy

        selected_cluster_accuracies = sorted(
            selected_cluster_accuracies.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        )
        selected_cluster_accuracies = {
            cluster_id: acc for cluster_id, acc in selected_cluster_accuracies
        }
        with open(args.output_selected_cluster_accuracies, "w") as f:
            json.dump(selected_cluster_accuracies, f, indent=4)
        selected_cluster_ids = [
            cluster_id for cluster_id in selected_cluster_accuracies.keys()
        ]

        print(f"Selected {len(selected_cluster_ids)} clusters")
        print(f"Total samples: {total_samples}")
        # avg_acc = sum(cluster_accuracies[cid]['accuracy'] for cid in selected_cluster_ids) / len(selected_cluster_ids)
        # print(f"Average accuracy of selected clusters: {avg_acc:.4f}")

        # Get the data for selected clusters
        allowed_cluster_ids = [int(cluster_id) for cluster_id in selected_cluster_ids]

        selected_data = []
        for sample in data:
            if sample["cluster_id"] in allowed_cluster_ids:
                selected_data.append(sample)
        # sort as per selected_cluster_accuracies
        # selected_data = sorted(selected_data, key=lambda x: selected_cluster_accuracies[str(x["cluster_id"])]['accuracy'], reverse=True)
        assert len(selected_data) == total_samples
        # Save results
        with open(args.output_file, "w") as f:
            json.dump(selected_data, f, indent=4)
        if os.path.exists(args.step_list_file):
            with open(args.step_list_file, "r") as f:
                step_list = [int(line.strip()) for line in f.readlines()]
        else:
            step_list = [391, 469]
        step_list.append((len(selected_data)) // (4 * 8 * 4) + step_list[-1])
        with open(args.step_list_file, "w") as f:
            for step in step_list[:-1]:
                f.write(f"{step}\n")
            f.write(f"{step_list[-1]}")
    elif args.selection_hardest:
        print("Selection hardest")
        # select the hardest clusters
        cluster_accuracies = acc_data["cluster_accuracies"]
        sorted_clusters = sorted(
            cluster_accuracies.items(), key=lambda x: x[1]["accuracy"]
        )
        selected_cluster_ids = []
        total_samples = 0
        data = json.load(open(args.selection_file, "r"))
        sample_num_per_cluster = {}
        for sample in data:
            if sample["cluster_id"] not in sample_num_per_cluster:
                sample_num_per_cluster[sample["cluster_id"]] = 1
            else:
                sample_num_per_cluster[sample["cluster_id"]] += 1
        # choose the hardest clusters
        while total_samples < args.selection_num:
            cluster_id = int(sorted_clusters[0][0])
            sorted_clusters.pop(0)
            if cluster_id not in sample_num_per_cluster:
                continue
            selected_cluster_ids.append(cluster_id)
            total_samples += sample_num_per_cluster[cluster_id]

        # get the accuracy of selected_cluster_ids
        # selected_cluster_ids = ['50', '49', '48', '47', '46', '45', '44', '43', '42', '41']
        selected_cluster_accuracies = {
            cluster_id: cluster_accuracies[str(cluster_id)]
            for cluster_id in selected_cluster_ids
        }
        # sort cluster ids by accuracy

        selected_cluster_accuracies = sorted(
            selected_cluster_accuracies.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        )
        selected_cluster_accuracies = {
            cluster_id: acc for cluster_id, acc in selected_cluster_accuracies
        }
        with open(args.output_selected_cluster_accuracies, "w") as f:
            json.dump(selected_cluster_accuracies, f, indent=4)
        selected_cluster_ids = [
            cluster_id for cluster_id in selected_cluster_accuracies.keys()
        ]

        print(f"Selected {len(selected_cluster_ids)} clusters")
        print(f"Total samples: {total_samples}")
        # avg_acc = sum(cluster_accuracies[cid]['accuracy'] for cid in selected_cluster_ids) / len(selected_cluster_ids)
        # print(f"Average accuracy of selected clusters: {avg_acc:.4f}")

        # Get the data for selected clusters
        allowed_cluster_ids = [int(cluster_id) for cluster_id in selected_cluster_ids]

        selected_data = []
        for sample in data:
            if sample["cluster_id"] in allowed_cluster_ids:
                selected_data.append(sample)
        assert len(selected_data) == total_samples
        # Save results
        with open(args.output_file, "w") as f:
            json.dump(selected_data, f, indent=4)
        if os.path.exists(args.step_list_file):
            with open(args.step_list_file, "r") as f:
                step_list = [int(line.strip()) for line in f.readlines()]
        else:
            step_list = [391, 469]
        step_list.append((len(selected_data)) // (4 * 8 * 4) + step_list[-1])
        with open(args.step_list_file, "w") as f:
            for step in step_list[:-1]:
                f.write(f"{step}\n")
            f.write(f"{step_list[-1]}")
    elif args.selection_easiest:
        print("Selection easiest")
        # select the easiest clusters
        cluster_accuracies = acc_data["cluster_accuracies"]
        sorted_clusters = sorted(
            cluster_accuracies.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True,  # Sort in descending order for highest accuracy first
        )
        selected_cluster_ids = []
        total_samples = 0
        data = json.load(open(args.selection_file, "r"))
        sample_num_per_cluster = {}
        for sample in data:
            if sample["cluster_id"] not in sample_num_per_cluster:
                sample_num_per_cluster[sample["cluster_id"]] = 1
            else:
                sample_num_per_cluster[sample["cluster_id"]] += 1
        sorted_clusters = [
            cluster
            for cluster in sorted_clusters
            if int(cluster[0]) in sample_num_per_cluster
        ]

        # choose the easiest clusters
        while total_samples < args.selection_num:
            cluster_id = int(sorted_clusters[0][0])
            sorted_clusters.pop(0)
            if cluster_id not in sample_num_per_cluster:
                continue
            selected_cluster_ids.append(cluster_id)
            total_samples += sample_num_per_cluster[cluster_id]

        # get the accuracy of selected_cluster_ids
        selected_cluster_accuracies = {
            cluster_id: cluster_accuracies[str(cluster_id)]
            for cluster_id in selected_cluster_ids
        }
        # sort cluster ids by accuracy
        selected_cluster_accuracies = sorted(
            selected_cluster_accuracies.items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        )
        selected_cluster_accuracies = {
            cluster_id: acc for cluster_id, acc in selected_cluster_accuracies
        }
        with open(args.output_selected_cluster_accuracies, "w") as f:
            json.dump(selected_cluster_accuracies, f, indent=4)
        selected_cluster_ids = [
            cluster_id for cluster_id in selected_cluster_accuracies.keys()
        ]

        print(f"Selected {len(selected_cluster_ids)} clusters")
        print(f"Total samples: {total_samples}")

        # Get the data for selected clusters
        allowed_cluster_ids = [int(cluster_id) for cluster_id in selected_cluster_ids]

        selected_data = []
        for sample in data:
            if sample["cluster_id"] in allowed_cluster_ids:
                selected_data.append(sample)
        assert len(selected_data) == total_samples
        # Save results
        with open(args.output_file, "w") as f:
            json.dump(selected_data, f, indent=4)
        if os.path.exists(args.step_list_file):
            with open(args.step_list_file, "r") as f:
                step_list = [int(line.strip()) for line in f.readlines()]
        else:
            step_list = [391, 469]
        step_list.append((len(selected_data)) // (4 * 8 * 4) + step_list[-1])   
        with open(args.step_list_file, "w") as f:
            for step in step_list[:-1]:
                f.write(f"{step}\n")
            f.write(f"{step_list[-1]}")
    elif args.selection_relative_change:
        # Calculate relative accuracy changes
        with open(args.output_average_acc_change, "r") as f:
            acc_change = json.load(f)
        relative_acc_change = acc_change["relative_acc_change"]

        # Calculate target samples per cluster based on proportional changes
        data = json.load(open(args.selection_file, "r"))
        # print("len of data in selection file", len(data))
        sample_num_per_cluster = {}
        for sample in data:
            if sample["cluster_id"] not in sample_num_per_cluster:
                sample_num_per_cluster[sample["cluster_id"]] = 1
            else:
                sample_num_per_cluster[sample["cluster_id"]] += 1
        print("samples per cluster in selection file", sample_num_per_cluster)

        # remove clusters from positive_changes if they have 0 samples
        # import pdb; pdb.set_trace()
        positive_changes = {k: v for k, v in relative_acc_change.items() if v > 0}
        positive_changes = {
            k: v
            for k, v in positive_changes.items()
            if int(k) in sample_num_per_cluster
        }
        print("positive_changes before softmax", positive_changes)
        if args.softmax_selection:
            # Apply softmax to the positive changes
            changes_values = list(positive_changes.values())
            softmax_proportions = softmax(changes_values, args.temperature)
            # Create new proportions dictionary
            positive_changes = dict(zip(positive_changes.keys(), softmax_proportions))
            total_positive_change = 1.0  # Softmax outputs already sum to 1
        else:
            total_positive_change = sum(positive_changes.values())
        target_samples_per_cluster = {}
        target_samples_per_cluster_ideal = {}
        print("positive_changes after softmax", positive_changes)
        for cluster_id, rel_change in positive_changes.items():
            proportion = rel_change / total_positive_change
            target_samples = int(args.selection_num * 0.90 * proportion)
            target_samples_per_cluster[int(cluster_id)] = min(
                target_samples, sample_num_per_cluster[int(cluster_id)]
            )
            target_samples_per_cluster_ideal[int(cluster_id)] = target_samples

        # Select samples according to calculated proportions
        # import pdb; pdb.set_trace()
        print("target_samples_per_cluster_ideal", target_samples_per_cluster_ideal)
        print("--------------------------------")
        print("target_samples_per_cluster", target_samples_per_cluster)

        print("--------------------------------")
        selected_data = []
        random.shuffle(data)
        for sample in data:
            cluster_id = sample["cluster_id"]
            if cluster_id in target_samples_per_cluster:
                if target_samples_per_cluster[cluster_id] > 0:
                    selected_data.append(sample)
                    target_samples_per_cluster[cluster_id] -= 1

        # Get samples from unselected clusters
        # import pdb; pdb.set_trace()
        print(f"Fastest selection: Selected {len(selected_data)} samples total")
        selected_cluster_ids = set(target_samples_per_cluster.keys())
        print("Selected cluster ids", selected_cluster_ids)
        unselected_samples = [
            sample
            for sample in data
            if sample["cluster_id"] not in selected_cluster_ids
        ]

        # Sample 10% from unselected clusters
        num_additional_samples = int(args.selection_num * 0.10)
        additional_samples = random.sample(unselected_samples, num_additional_samples)
        selected_data.extend(additional_samples)
        # import pdb; pdb.set_trace()
        print(f"Selected {len(selected_data)} samples total")

        print(f"Including {len(additional_samples)} samples from unselected clusters")


        print("Iterations to run: ", (len(selected_data)) / (2 * 16 * 4))
        print("we use (2 * 16 * 4) to calculate iterations to run")
        with open(args.output_file, "w") as f:
            json.dump(selected_data, f, indent=4)

        # save selected cluster accuracies
        with open(args.output_selected_cluster_accuracies, "w") as f:
            json.dump(positive_changes, f, indent=4)

        if os.path.exists(args.step_list_file):
            with open(args.step_list_file, "r") as f:
                step_list = [int(line.strip()) for line in f.readlines()]
        else:
            step_list = [391, 469]
        step_list.append((len(selected_data)) // (2 * 16 * 4) + step_list[-1])
        with open(args.step_list_file, "w") as f:
            for step in step_list[:-1]:
                f.write(f"{step}\n")
            f.write(f"{step_list[-1]}")



def softmax(x, temperature=1.0):
    """Compute softmax values with temperature."""
    x = np.array(list(x))
    x = x / temperature
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()


if __name__ == "__main__":
    args = parse_args()
    main(args)
