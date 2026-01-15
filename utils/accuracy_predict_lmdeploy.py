import torch
from tqdm import tqdm
import argparse
import os

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

os.environ["PYTHONHASHSEED"] = str(seed)
from lmdeploy import (
    pipeline,
    TurbomindEngineConfig,
    ChatTemplateConfig,
    GenerationConfig,
)
from tqdm import tqdm
import json
import random
from lmdeploy.vl import load_image



@torch.no_grad()
def pred_accuracy(pipe, args, split_list, temp_results_file, temp_results):
    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens)

    os.makedirs(os.path.dirname(args.acc_output_file), exist_ok=True)
    batch_size = args.perplexity_batch_size
    step = 0
    for i in tqdm(
        range(0, len(split_list), batch_size), total=(len(split_list) + batch_size - 1) // batch_size
    ):
        batch = split_list[i : i + batch_size]
        # Continue with the rest of the processing
        question_list = []
        answer_list = []
        cluster_id_list = []
        multimodal_prompts = []
        unimodal_prompts = []
        is_multimodal_list = []
        for item in batch:

            question = item["conversations"][0]["value"]

            if "image" in item:
                image = load_image(os.path.join(args.image_folder, item["image"]))
                multimodal_prompts.append((question, image))
                is_multimodal_list.append(True)
            else:
                unimodal_prompts.append(question)
                is_multimodal_list.append(False)
            question_list.append(question)
            answer_list.append(item["conversations"][1]["value"])
            if "cluster_id" in item:
                cluster_id_list.append(item["cluster_id"])
            else:
                cluster_id_list.append(item["internVL_decision"])
        generated_answers = []
        if multimodal_prompts:
            multimodal_outputs = pipe(multimodal_prompts, gen_config=gen_config)
            if isinstance(multimodal_outputs, list):
                multimodal_answers = [output.text for output in multimodal_outputs]
            else:
                multimodal_answers = [multimodal_outputs.text]
        else:
            multimodal_answers = []

        if unimodal_prompts:
            unimodal_outputs = pipe(unimodal_prompts, gen_config=gen_config)
            if isinstance(unimodal_outputs, list):
                unimodal_answers = [output.text for output in unimodal_outputs]
            else:
                unimodal_answers = [unimodal_outputs.text]
        else:
            unimodal_answers = []
        # Merge answers back in the correct order
        current_multimodal_idx = 0
        current_unimodal_idx = 0
        for is_multimodal in is_multimodal_list:
            if is_multimodal:
                generated_answers.append(multimodal_answers[current_multimodal_idx])
                current_multimodal_idx += 1
            else:
                generated_answers.append(unimodal_answers[current_unimodal_idx])
                current_unimodal_idx += 1
        temp_results["predicted_answers"].extend(generated_answers)
        temp_results["correct_answers"].extend(answer_list)
        temp_results["questions"].extend(question_list)
        temp_results["cluster_ids"].extend(cluster_id_list)
        step += 1
        if step % 50 == 1:
            with open(temp_results_file, "w") as f:
                json.dump(temp_results, f)
    # check still have data in the last
    if len(temp_results["predicted_answers"]) < len(split_list):
        print(f"Still have data in the last batch")
        last_batch_size = len(split_list) - len(temp_results["predicted_answers"])
        batch = split_list[-last_batch_size:]
        question_list = []
        answer_list = []
        cluster_id_list = []
        multimodal_prompts = []
        unimodal_prompts = []
        is_multimodal_list = []
        for item in batch:
            question = item["conversations"][0]["value"]
            if "image" in item:
                image = load_image(os.path.join(args.image_folder, item["image"]))
                multimodal_prompts.append((question, image))
                is_multimodal_list.append(True)
            else:
                unimodal_prompts.append(question)
                is_multimodal_list.append(False)
            question_list.append(question)
            answer_list.append(item["conversations"][1]["value"])
            cluster_id_list.append(item["cluster_id"])
        generated_answers = []
        if multimodal_prompts:
            multimodal_outputs = pipe(multimodal_prompts, gen_config=gen_config)
            if isinstance(multimodal_outputs, list):
                multimodal_answers = [output.text for output in multimodal_outputs]
            else:
                multimodal_answers = [multimodal_outputs.text]
        else:
            multimodal_answers = []
        if unimodal_prompts:
            unimodal_outputs = pipe(unimodal_prompts, gen_config=gen_config)
            if isinstance(unimodal_outputs, list):
                unimodal_answers = [output.text for output in unimodal_outputs]
            else:
                unimodal_answers = [unimodal_outputs.text]
        else:
            unimodal_answers = []
        current_multimodal_idx = 0
        current_unimodal_idx = 0
        for is_multimodal in is_multimodal_list:
            if is_multimodal:
                generated_answers.append(multimodal_answers[current_multimodal_idx])
                current_multimodal_idx += 1
            else:
                generated_answers.append(unimodal_answers[current_unimodal_idx])
                current_unimodal_idx += 1
        temp_results["predicted_answers"].extend(generated_answers)
        temp_results["correct_answers"].extend(answer_list)
        temp_results["questions"].extend(question_list)
        temp_results["cluster_ids"].extend(cluster_id_list)

    with open(temp_results_file, "w") as f:
        json.dump(temp_results, f)

    # Create final results dictionary using data from temp file
    with open(temp_results_file, "r") as f:
        temp_results = json.load(f)

    results_dict = {
        "predicted_answers": temp_results["predicted_answers"],
        "correct_answers": temp_results["correct_answers"],
        "cluster_ids": temp_results["cluster_ids"],
        "questions": temp_results["questions"],
    }

    # Write final results
    with open(args.acc_output_file, "w") as pred_file:
        json.dump(results_dict, pred_file, indent=4)


def eval_model(args):
    # Model
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    pipe = pipeline(
        args.model_name_or_path,
        backend_config=TurbomindEngineConfig(
            cache_max_entry_count=0.9, session_len=8192, tp=torch.cuda.device_count()
        ),
        chat_template_config=ChatTemplateConfig(model_name="vicuna"),
    )
    data = json.load(open(args.data_path))
    total_samples_per_chunk = len(data) // args.num_chunks

    split_list = data[
        args.chunk_idx
        * total_samples_per_chunk : (args.chunk_idx + 1)
        * total_samples_per_chunk
    ]
    if args.chunk_idx == args.num_chunks - 1:
        split_list = data[
            args.chunk_idx * total_samples_per_chunk :
        ]
    # Create a temporary file for batch results
    temp_results_file = args.acc_output_file + ".temp"
    # Initialize or load existing results
    if os.path.exists(temp_results_file):
        with open(temp_results_file, "r") as f:
            temp_results = json.load(f)
            # Resume from existing results
            # Skip processed batches
            resume_index = len(temp_results.get("predicted_answers", []))
            split_list = split_list[resume_index:]
            print(f"Resuming from index: {resume_index}")
    else:
        temp_results = {
            "predicted_answers": [],
            "correct_answers": [],
            "cluster_ids": [],
            "questions": [],
            "": [],
        }

    print(f"total_samples_per_chunk: {total_samples_per_chunk}")
    pred_accuracy(pipe, args, split_list, temp_results_file, temp_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/home/mila/q/qian.yang/scratch/visual_imagination/llava_lora/llava_lora_warmup_60K_total_133k_1K_clusters_AllSelection_Fastest_Soft_0.5_Prev_random_acc_663_merged",
    )
    parser.add_argument(
        "--specific_cluster_file",
        type=str,
        default=None,
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/network/scratch/q/qian.yang/llava-v1.5-7b/instruct_tuning_data/data",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/mila/q/qian.yang/scratch/llava-v1.5-7b/instruct_tuning_data/All_LBA_1K_7K_Soft_0.5_Prev_efficient_reorder/llava_v1_5_mix665k_dino-bert_1K_Top_60K_until_1024_random_100.json",
    )
    parser.add_argument(
        "--acc_output_file",
        type=str,
        default="./out_acc.json",
    )
    parser.add_argument("--max_new_tokens", type=int, default=25)
    parser.add_argument("--perplexity_batch_size", type=int, default=100)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    args = parser.parse_args()
    eval_model(args)
