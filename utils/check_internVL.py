import json
from tqdm import tqdm as tqdm
import argparse
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
import torch
# from llava.utils import disable_torch_init
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
torch.backends.cudnn.benchmark = False     # ensures deterministic behavior
# Optional: Set environment variable for any libraries that check it
import os
os.environ['PYTHONHASHSEED'] = str(seed)

oracle_model = pipeline(
    "OpenGVLab/InternVL2-26B",
    backend_config=TurbomindEngineConfig(
        dtype="float16", cache_max_entry_count=0.5, session_len=8192, tp=torch.cuda.device_count()
    ),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Check answers using InternVL model")
    parser.add_argument(
        "--initial_path",
        type=str,
        default="/home/mila/c/chandhos/scratch/checkpoints/llava_lora_tinyllava_ours_prune_0.2_dino-bert_v1.5_warmup_c1_recluster_mid_stage1/selection_step_1039",
        help="Path to the directory containing the json file to process",
    )
    parser.add_argument(
        "--final_path",
        type=str,
        default=None,
        help="Path to the directory containing the json file to process",
    )
    parser.add_argument(
        "--output_selected_cluster_accuracies",
        type=str,
        default=None,
        help="Path to the file containing the selected cluster accuracies",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to the file containing the selected cluster accuracies",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default=None,
        help="Path to the file containing the selected cluster accuracies",
    )
    parser.add_argument("--bs", type=int, default=4, help="Batch size")
    return parser.parse_args()


def get_prompt(q1, q2, q3):
    CAPTION_PROMPT = (
        """For the given "Question", there is a "predicted_answer" and a "ground_truth_answer". Check if the "predicted_answer" is correct answer for the given question.
    Give reason for your prediction.

    OUTPUT Format-
    Answer - correct or incorrect
    Reason - 

    Question -"""
        + str(q1)
        + """\npredicted_answer -"""
        + str(q2)
        + """\n ground_truth_answer -"""
        + str(q3)
    )

    CAPTION_PROMPT_v2 ="""Given a input question and two answers: a candidate answer and a reference answer, determine if the candidate answer is correct or incorrect.
    Rules:
    - The candidate answer is correct if it is semantically equivalent to the reference answer, even if they are phrased differently
    - The candidate answer should be marked as incorrect if it:
    * Contains factual errors compared to reference answer
    * Only partially answers the question
    * Includes hedging language (e.g., "probably", "likely", "I think", etc.)
    * Answers a different question than what was asked
    - Give a reason for your prediction.
    Output "correct" if the candidate answer is correct, or "incorrect" if it is incorrect.
    OUTPUT Format-
    Answer - correct or incorrect
    Reason - 

    INPUT-
    Question""" + str(q1) +"""\nCandidate Answer:""" + str(q2)+"""\nReference Answer:""" + str(q3)
    return CAPTION_PROMPT_v2

def main():
    # disable_torch_init()
    args = parse_args()
    if args.final_path is None:
        output_dict = {'questions':[],'predicted_answers':[],'correct_answers':[],'cluster_id':[], 'internVL_decision':[]}
        file_paths = [
            os.path.join(args.initial_path, args.file_name),
            
        ]
        resp = []
        answer = []
        answer_list = []
        for file_path in file_paths:
            with open(file_path) as f:
                print('Reading file: ', file_path)
                acc_dict = json.load(f)
            

            total_iter = (len(acc_dict['questions']) + args.bs - 1) // args.bs
            print(f"Total iterations: {total_iter}")
            for i in tqdm(range(0, len(acc_dict['questions']), args.bs)):
                print(f"Processing batch {i//args.bs + 1} of {total_iter}")
                q1 = acc_dict['questions'][i:i+args.bs]
                q2 = acc_dict['predicted_answers'][i:i+args.bs]
                q3 = acc_dict['correct_answers'][i:i+args.bs]
        

                #[(CAPTION_PROMPT, load_image(img_url)) for img_url in path]
                prompts = [(get_prompt(q1, q2, q3)) for q1, q2, q3 in zip(q1, q2, q3)]
                #import pdb; pdb.set_trace()
                response = oracle_model(prompts)
                
                answer_list = []
                for respo in response:
                    ans = respo.text.split("Reason -", 1)[0].replace("Answer -", "").strip()
                    if ans.lower() in ['correct']:
                        answer_list.append(1)
                    elif ans.lower() in ['incorrect']:
                        answer_list.append(0)
                    else:
                        print('oops')
                        answer_list.append(0)
                output_dict['questions'].extend(q1)
                output_dict['predicted_answers'].extend(q2)
                output_dict['correct_answers'].extend(q3)
                output_dict['internVL_decision'].extend(answer_list)
                output_dict['cluster_id'].extend(acc_dict['cluster_ids'][i:i+args.bs])
                
                #import pdb; pdb.set_trace()
            
        
        
        #calculate cluster accuracies
        print('length of cluster_id: ', len(output_dict['cluster_id']))
        print('length of internVL_decision: ', len(output_dict['internVL_decision']))
        cluster_acc = {}
        for i in range(len(output_dict['cluster_id'])):
            if output_dict['cluster_id'][i] not in cluster_acc:
                cluster_acc[output_dict['cluster_id'][i]] = {'correct':0, 'total':0, 'accuracy':0}
            if output_dict['internVL_decision'][i] == 1:
                cluster_acc[output_dict['cluster_id'][i]]['correct'] += 1
            cluster_acc[output_dict['cluster_id'][i]]['total'] += 1

        for cluster_id in cluster_acc:
            cluster_acc[cluster_id]['accuracy'] = cluster_acc[cluster_id]['correct'] / cluster_acc[cluster_id]['total']

        acc_dict = {}
        acc_dict['cluster_accuracies'] = cluster_acc
        acc_dict['total_samples'] = len(output_dict['internVL_decision'])
        acc_dict['total_cluster_samples'] = sum(cluster_acc[cluster_id]['total'] for cluster_id in cluster_acc)
        acc_dict['total_correct_samples'] = sum(cluster_acc[cluster_id]['correct'] for cluster_id in cluster_acc)
        acc_dict['overall_accuracy'] = acc_dict['total_correct_samples'] / acc_dict['total_samples']
        with open(os.path.join(args.initial_path, args.file_name.split(".")[0]+"_internVL_decision.json"), "w") as json_file:
            json.dump(output_dict, json_file, indent=4)
            
        with open(os.path.join(args.initial_path, args.output_file), "w") as json_file:
            json.dump(acc_dict, json_file, indent=4)

    #calculate change in cluster accuracy across two file

    #load two files
    if args.final_path is not None:
        file1 = os.path.join(args.initial_path, "combined_results_internVL.json")
        file2 = os.path.join(args.final_path, "combined_results_internVL.json")

        with open(file1) as f:
            data1 = json.load(f)
        with open(file2) as f:
            data2 = json.load(f)
        if args.output_selected_cluster_accuracies is not None:
            with open(args.output_selected_cluster_accuracies, "r") as f:
                selected_cluster_accuracies = json.load(f)
            selected_cluster_ids = [
                cluster_id for cluster_id in selected_cluster_accuracies.keys()
            ]
        else:
            selected_cluster_ids = data1['cluster_accuracies'].keys()
        #calculate change in cluster accuracy across two files
        change_dict = {}
        for cluster_id in selected_cluster_ids:
            if cluster_id in data2['cluster_accuracies']:
                change = data2['cluster_accuracies'][cluster_id]['accuracy'] - data1['cluster_accuracies'][cluster_id]['accuracy']
                change_dict[cluster_id] = change

    #sort change_dict by value
        change_dict = dict(sorted(change_dict.items(), key=lambda item: item[1], reverse=True))
        with open(os.path.join(args.initial_path, "change_dict_final.json"), "w") as json_file:
            json.dump(change_dict, json_file, indent=4)

    


if __name__ == "__main__":
    main()
                
