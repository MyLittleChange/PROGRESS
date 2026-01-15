import os
import pathlib
import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse

# Load a pre-trained Sentence-Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


def main(args):
    path = args.path
    output_path = args.output_path
    if path.endswith(".jsonl"):
        with open(path) as f:
            candidates = [json.loads(line) for line in f]
    else:
        with open(path) as f:
            candidates = json.load(f)
    new_candidates = []
    features = []
    if 'TDIUC' in path:
        candidates=candidates['questions']
        for item in range(len(candidates)):
            candidates[item]['text']=candidates[item]['question']
    for i, anno in tqdm(enumerate(candidates), total=len(candidates)):
        split_lists = ["\nAnswer with", "\nAnswer the"]
        if "text" in anno:
            con_str = anno["text"]
        elif "question" in anno:
            print(anno["question"])
            con_str = anno["question"]
        else:
            con_str = "\n".join(
                x["value"] for x in anno["conversations"] if x["from"] == "human"
            )
        for split_list in split_lists:
            if split_list in con_str:
                con_str = con_str.split(split_list)[0]
                break
        new_candidates.append(con_str)
        features.append(model.encode(con_str))

    np.save(
        output_path,
        features,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
