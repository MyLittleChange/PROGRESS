import torch
from tqdm import tqdm
import argparse
import torch
import os
from tqdm import tqdm
from train import make_supervised_data_module
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    get_model_name_from_path,
)
from torch.utils.data import DataLoader
import json
import torch.nn.functional as F


def get_loss_per_question(logits, processed_labels, attention_mask):
    # TODO: implement different score functions
    logits = logits[:, :-1]
    processed_labels = processed_labels[:, 1:]
    cross_entropy_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        processed_labels.reshape(-1),
        reduction="none",
    ).reshape(logits.size(0), -1)
    loss_per_question = torch.sum(cross_entropy_loss, dim=1) / attention_mask.sum(dim=1)
    return loss_per_question.cpu().tolist()


@torch.no_grad()
def pred_confidence(
    model, training_args, dataloader, tokenizer, temp_results_file, temp_results
):
    model.eval()
    model.to(dtype=torch.float16).to("cuda")

    os.makedirs(os.path.dirname(training_args.acc_output_file), exist_ok=True)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        cluster_ids = batch.pop("cluster_ids")
        # We are calculating the confidence score using all question,
        # rightnow we directly use the loss (cross entropy loss) as the confidence score
        global_indexs = batch.pop("indexs")
        batch["images"] = batch["images"].to(dtype=torch.float16).to(model.device)
        batch["input_ids"] = batch["input_ids"].to(model.device)
        batch["attention_mask"] = batch["attention_mask"].to(model.device)
        batch["labels"] = batch["labels"].to(model.device)
        outputs = model(**batch, return_dict=True, return_processed_lab=True)

        logits = outputs.logits
        processed_labels = outputs["processed_label"]
        confidence_scores = get_loss_per_question(
            logits, processed_labels, batch["attention_mask"]
        )
        temp_results["cluster_ids"].extend(cluster_ids)
        temp_results["confidence_scores"].extend(confidence_scores)
        if i % 200 == 1:
            with open(temp_results_file, "w") as f:
                json.dump(temp_results, f)
    # write the last batch results
    with open(temp_results_file, "w") as f:
        json.dump(temp_results, f)

    # Create final results dictionary using data from temp file
    with open(temp_results_file, "r") as f:
        temp_results = json.load(f)

    results_dict = {
        "confidence_scores": temp_results["confidence_scores"],
        "cluster_ids": temp_results["cluster_ids"],
    }

    # Write final results
    with open(training_args.acc_output_file, "w") as pred_file:
        json.dump(results_dict, pred_file, indent=4)


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_name_or_path)
    if "lora" in args.model_name_or_path:
        model_name = get_model_name_from_path(model_path) + "_llava_lora"
    else:
        model_name = get_model_name_from_path(model_path) + "_llava"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    model.eval()
    args.uncertainty_selection = False
    args.uncertainty_selection_oracle = True
    args.local_clip_selection = False
    args.selection_batch_size = 0
    args.chosen_index_output_file = None
    args.max_seq_len = 2048
    args.image_processor = image_processor
    args.mm_use_im_start_end = False
    args.mm_use_im_patch_token = False
    args.is_multimodal = True
    args.first_question_only = False
    args.random_question = False
    args.reverse_padding = False
    args.get_cluster_ids = True
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=args,
        training_args=args,
    )
    total_samples_per_chunk = len(data_module["train_dataset"]) // args.num_chunks
    split_list = data_module["train_dataset"].list_data_dict[
        args.chunk_idx
        * total_samples_per_chunk : (args.chunk_idx + 1)
        * total_samples_per_chunk
    ]
    # Create a temporary file for batch results
    temp_results_file = args.acc_output_file + ".temp"
    # Initialize or load existing results
    if os.path.exists(temp_results_file):
        with open(temp_results_file, "r") as f:
            temp_results = json.load(f)
            # Resume from existing results
            # Skip processed batches
            resume_index = len(temp_results.get("cluster_ids", []))
            split_list = split_list[resume_index:]
            print(f"Resuming from index: {resume_index}")
    else:
        temp_results = {
            "cluster_ids": [],
            "confidence_scores": [],
        }

    data_module["train_dataset"].list_data_dict = split_list
    # start from the last index of the existing_indexs
    print(f"total_samples_per_chunk: {total_samples_per_chunk}")
    dataloader = DataLoader(
        data_module["train_dataset"],
        batch_size=args.perplexity_batch_size,
        shuffle=False,
        collate_fn=data_module["data_collator"],
    )
    pred_confidence(model, args, dataloader, tokenizer, temp_results_file, temp_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/network/scratch/q/qian.yang/visual_imagination/llava_lora/llava_lora_warmup_133k_total_200k_coincide_bert",
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
        default="/network/scratch/q/qian.yang/llava-v1.5-7b/instruct_tuning_data/llava_v1_5_mix665k_with_COINCIDE_bert.json",
    )
    parser.add_argument(
        "--acc_output_file",
        type=str,
        default="./out_acc.json",
    )
    parser.add_argument("--random_sample_per_cluster", type=bool, default=False)
    parser.add_argument("--centroid_sample_per_cluster", type=bool, default=False)
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--perplexity_batch_size", type=int, default=8)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--first_question_only", type=bool, default=False)
    parser.add_argument("--random_question", type=bool, default=False)
    parser.add_argument("--all_questions_separate", type=bool, default=False)
    args = parser.parse_args()

    eval_model(args)
