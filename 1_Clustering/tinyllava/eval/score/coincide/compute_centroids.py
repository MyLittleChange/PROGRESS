import argparse
import random
import numpy as np
import os
import logging
from tinyllava.eval.score.coincide.clustering import compute_centroids
from packaging import version
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
from PIL import Image

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="none")
    parser.add_argument(
        "--llava_data_file",
        type=str,
        required=True,
        help="Path to the data JSON file",
    )
    parser.add_argument("--specific_indices_file", type=str, default=None)
    parser.add_argument("--remove_indices_path", type=str, default=None)
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output JSON file",
    )
    parser.add_argument(
        "--dino_features_path",
        type=str,
        default=None,
        help="Path to DINO features (.npy file)",
    )
    parser.add_argument(
        "--bert_features_path",
        type=str,
        default=None,
        help="Path to BERT/question features (.npy file)",
    )
    parser.add_argument("--sim_metric", type=str, default="cosine")
    parser.add_argument("--Kmeans_with_cos_dist", action="store_true")
    parser.add_argument("--emb_memory_loc", type=str, default="emb.npy")
    parser.add_argument("--save_folder", type=str, default="./save_folder")
    parser.add_argument(
        "--ncentroids", type=int, default=500
    )  # proportional to dataset size
    parser.add_argument("--niter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="whether to visualize the clustering results",
    )
    parser.add_argument("--vision_flan", action='store_true')
    parser.add_argument("--llava_next", action='store_true')
    parser.add_argument("--llava_flan", action='store_true')
    parser.add_argument("--icons", action='store_true')
    parser.add_argument("--base_dir", type=str, default=None)
    args = parser.parse_args()

    ## -- Fix the seed
    SEED = args.seed
    random.seed(SEED)

    # Load features based on feature_type
    if args.feature_type == "dino":
        if args.dino_features_path is None:
            raise ValueError("--dino_features_path is required when feature_type is 'dino'")
        print("Using DINO features")
        emb_memory = np.load(args.dino_features_path)
    elif args.feature_type == "bert":
        if args.bert_features_path is None:
            raise ValueError("--bert_features_path is required when feature_type is 'bert'")
        print("Using BERT features")
        emb_memory = np.load(args.bert_features_path)
    elif args.feature_type == "dino-bert":
        if args.dino_features_path is None or args.bert_features_path is None:
            raise ValueError("--dino_features_path and --bert_features_path are required when feature_type is 'dino-bert'")
        print("Using DINO-BERT features")
        features_dino = np.load(args.dino_features_path)
        features_bert = np.load(args.bert_features_path)
        emb_memory = np.concatenate([features_dino, features_bert], axis=-1)
    else:
        print("Using Original COINCIDE features")
        emb_memory = np.load(args.emb_memory_loc)
    with open(args.llava_data_file, "r") as f:
            data = json.load(f)
    if args.specific_indices_file is not None and args.remove_indices_path is not None:
        # both are not None, need to first remove the indices and then use the specific indices
        remove_indices = np.load(args.remove_indices_path)
        emb_memory = np.delete(emb_memory, remove_indices, axis=0)
        data = [data[i] for i in range(len(data)) if i not in remove_indices]
        print("Data shape after remove indices", len(data))
        print("Emb memory shape after remove indices", emb_memory.shape)
        specific_indices = np.load(args.specific_indices_file)
        emb_memory = emb_memory[specific_indices]
        data = [data[i] for i in specific_indices]
        print("Data shape after specific indices", len(data))
        print("Emb memory shape after specific indices", emb_memory.shape)
    elif args.specific_indices_file is not None:
        # load llava data
        # load json data
        specific_indices = np.load(args.specific_indices_file)
        print("Using specific indices", specific_indices.shape)
        emb_memory = emb_memory[specific_indices]

        data = [data[i] for i in specific_indices]
        print("Data shape after specific indices", len(data))
        print("Emb memory shape after specific indices", emb_memory.shape)

    elif args.remove_indices_path is not None:

        remove_indices = np.load(args.remove_indices_path)
        emb_memory = np.delete(emb_memory, remove_indices, axis=0)
        data = [data[i] for i in range(len(data)) if i not in remove_indices]
        print("Data shape after remove indices", len(data))
        print("Emb memory shape after remove indices", emb_memory.shape)
    # Normalize since SemDeDup uses Spherical Kmeans clustering with normalized embeddings, referring to paper, even in language modality with OPT model.
    dataset_size, emb_size = emb_memory.shape
    emb_memory = emb_memory / np.linalg.norm(emb_memory, axis=-1, keepdims=True)
    print("Emb memory shape after normalization", emb_memory.shape)

    compute_centroids(
        data=emb_memory,
        ncentroids=args.ncentroids,
        niter=args.niter,
        seed=args.seed,
        Kmeans_with_cos_dist=args.Kmeans_with_cos_dist,
        save_folder=args.save_folder,
        logger=logger,
        verbose=True,
    )