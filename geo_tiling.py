import math
import os
import random
from copy import deepcopy
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from osgeo import gdal
from tqdm import tqdm

# SONY RX-1 at 270m height
WINDOW_WIDTH = 276.1714286
WINDOW_HEIGHT = 183.6
FPS = 25


def get_center_xy(ds):
    width = ds.RasterXSize
    height = ds.RasterYSize
    (
        minx,
        pixel_width_meters,
        x_rotation,
        maxy,
        y_rotation,
        cell_height_meters,
    ) = ds.GetGeoTransform()
    maxx = minx + width * pixel_width_meters + height * x_rotation
    miny = maxy + width * y_rotation + height * cell_height_meters
    return (minx + maxx) / 2, (miny + maxy) / 2


def shift(x: float, y: float, v_min: float, v_max: float) -> Tuple[float, float]:
    """random coordinate shift. For cartesian systems only."""
    v = random.uniform(v_min, v_max)
    dx = random.uniform(-v, v)
    dy = math.sqrt(v**2 - dx**2)
    if random.random() > 0.5:
        dy = -dy
    return x + dx, y + dy


def geo_crop(
    ds: gdal.Dataset,
    center_x: float,
    center_y: float,
    window_x: float = WINDOW_WIDTH,
    window_y: float = WINDOW_HEIGHT,
    destination_path: str = "/vsimem/crop.tif",
) -> gdal.Dataset:
    """Crop ds by center and size args
    center_x, center_y - center coordinates
    window_x, window_y - window side lengths in same coordinate system as center_x, center_y
    """
    ulx = center_x - window_x / 2
    uly = center_y + window_y / 2
    brx = center_x + window_x / 2
    bry = center_y - window_y / 2
    window = (ulx, uly, brx, bry)
    return gdal.Translate(destName=destination_path, srcDS=ds, projWin=window)


def is_valid(img: np.ndarray) -> bool:
    """Check if img contains nodata pixels.
    Nodata is determined as zero in all channels.
    CHW order only"""
    return not np.any(img.sum(axis=0) == 0)


def imwrite(dest: str, ds: gdal.Dataset, side_limit=6000) -> float:
    img = ds.ReadAsArray()
    img = img[:-1].swapaxes(0, -1)  # drop alpha channel
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    r = min(1, side_limit / max(img.shape))
    if r < 1:
        img = cv2.resize(
            img, dst=None, dsize=None, fx=r, fy=r, interpolation=cv2.INTER_AREA
        )
    cv2.imwrite(dest, img)
    return r


def generate_samples(
    ds: gdal.Dataset,
    destination_directory: str,
    n: Optional[int] = 1000,
    v_min: float = 0,
    v_max: float = 10 / FPS,
    window_width: float = WINDOW_WIDTH,
    window_height: float = WINDOW_HEIGHT,
    max_tries: int = 10,
) -> dict:
    """Generate n samples from ds center with random shift"""
    images_dir = os.path.join(destination_directory, "images")
    os.makedirs(images_dir, exist_ok=True)
    center_x, center_y = get_center_xy(ds)
    _, pixel_width_meters, _, _, _, pixel_height_meters = ds.GetGeoTransform()
    meta_path = os.path.join(destination_directory, "meta.csv")
    meta = {
        "meters_in_pixel_x": [pixel_width_meters],
        "meters_in_pixel_y": [pixel_height_meters]
    }
    pd.DataFrame(meta).to_csv(meta_path, index=None)
    result = []
    for i in tqdm(range(n)):
        success = False
        tries = 0
        while tries < max_tries and not success:
            nx, ny = shift(center_x, center_y, v_min, v_max)
            candidate_ds = geo_crop(nx, ny, window_width, window_height)
            candidate_img = candidate_ds.ReadAsArray()
            success = is_valid(candidate_img)
        if not success:
            print("Early stop: too much failed takes to crop proper image")
            return result
        r = imwrite(os.path.join(images_dir, f"{i}.png"), candidate_ds)
        shift_x = math.ceil(r * (nx - center_x) / pixel_width_meters)
        shift_y = math.ceil(r * (ny - center_y) / pixel_height_meters)
        center_x, center_y = nx, ny
        result.append(
            {
                "meter_center_x": center_x,
                "meter_center_y": center_y,
                "pixel_shift_x": shift_x if i else 0,
                "pixel_shift_y": shift_y if i else 0,
            }
        )
    return result


def main(src_path: str, destination_directory: str):
    shift_path = os.path.join(destination_directory, "centers.csv")
    ds = gdal.Open(src_path)
    result = generate_samples(ds, destination_directory)
    pd.DataFrame(result).to_csv(shift_path)


def gdal_error_handler(err_class, err_num, err_msg):
    errtype = {
            gdal.CE_None:'None',
            gdal.CE_Debug:'Debug',
            gdal.CE_Warning:'Warning',
            gdal.CE_Failure:'Failure',
            gdal.CE_Fatal:'Fatal'
    }
    err_msg = err_msg.replace('\n',' ')
    err_class = errtype.get(err_class, 'None')
    print ('Error Number: %s' % (err_num))
    print ('Error Type: %s' % (err_class))
    print ('Error Message: %s' % (err_msg))

if __name__=='__main__':

    # install error handler
    gdal.PushErrorHandler(gdal_error_handler)
    src_path = "meters.tif"
    dst = "geo_sampled_270"
    main(src_path, dst)
