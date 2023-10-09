import math
import os
from pathlib import Path

import cv2
import imreg_dft as ird
import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics import qa_csv


def imread(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img[:30, :30]


if __name__ == "__main__":
    image_dir = "pixel_shift_estimation/geo_sampled_270_fps/images"
    shift_dst = image_dir.replace("images", "ird_30x30.csv")
    paths = sorted(os.listdir(image_dir), key=lambda x: int(Path(x).stem))
    paths = [os.path.join(image_dir, path) for path in paths]

    shift = {"pixel_shift_x": [0], "pixel_shift_y": [0]}
    prev = imread(paths[0])
    for path in tqdm(paths[1:]):
        curr = imread(path)
        x, y = ird.translation(prev, curr)["tvec"]
        shift["pixel_shift_x"].append(math.ceil(x))
        shift["pixel_shift_y"].append(math.ceil(y))
        prev = curr

    df = pd.DataFrame(shift)
    df.to_csv(shift_dst, sep=",", index=None)
    qa_csv(
        gt_path=image_dir.replace("images", "centers.csv"),
        pred_path=shift_dst,
        meta_path=image_dir.replace("images", "meta.csv"),
        dst_path=shift_dst.replace(".csv", "_bench.csv")
    )
