# import torch
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from pathlib import Path
# from einops import rearrange, repeat
# from jaxtyping import Float
# from torch import Tensor


# file_path = Path("/storage/user/sade/mvsplat/datasets/re10k_full/test/000002.torch")
# # re_file = Path("/storage/user/sade/hypersim/re10k_first.torch")
# tensor = torch.load(file_path)


# print("2.torch")
# print("---------------------")
# for i in range(len(tensor)):
#     print(tensor[i]["key"])

import json
from pathlib import Path

import torch
from tqdm import tqdm

# DATASET_PATH = Path.home() / "mvsplat/datasets/aria"
DATASET_PATH = Path("/storage/user/sade/mvsplat/datasets/aria")

if __name__ == "__main__":
    for stage in DATASET_PATH.iterdir():
        index = {}
        for chunk_path in tqdm(list(stage.iterdir()), desc=f"Indexing {stage.name}"):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage))
        with (stage / "index.json").open("w") as f:
            json.dump(index, f)