from typing import Dict, Union

import numpy as np
import pandas as pd


def to_coord(x: float, y: float) -> np.array:
    return np.array([x, y])


def distance(a: np.array, b: np.array) -> Union[float, np.ndarray]:
    assert a.ndim == b.ndim
    return np.linalg.norm(a - b, axis=-1)


def pixel_error_per_meter(
    a_meter: np.array,
    b_meter: np.array,
    real_pix_shift: np.array,
    pred_pix_shift: np.array,
) -> Union[float, np.ndarray]:
    """How much pixel errors made per meter of real shift?"""
    shift_meter = distance(a_meter, b_meter)
    pix_error = distance(real_pix_shift, pred_pix_shift)
    return pix_error / shift_meter


def pixel_to_meter(pix: np.array, meters_in_pixel: np.array) -> np.ndarray:
    return pix * meters_in_pixel


def meter_error(
    real_pix: np.array,
    pred_pix: np.array,
    meters_in_pixel: np.array,
    relative: bool = False,
) -> Union[float, np.ndarray]:
    real_meter = pixel_to_meter(real_pix, meters_in_pixel)
    pred_meter = pixel_to_meter(pred_pix, meters_in_pixel)
    error = distance(real_meter, pred_meter)
    if relative:
        error /= np.linalg.norm(real_meter)
    return error


def qa_np(
    origin_meters: np.ndarray,
    real_pix_shift: np.ndarray,
    pred_pix_shift: np.ndarray,
    meters_in_pixel: np.ndarray,
) -> Dict[str, np.ndarray]:
    pixel_error = distance(real_pix_shift, pred_pix_shift)
    ame = meter_error(real_pix_shift, pred_pix_shift, meters_in_pixel, relative=False)
    rme = meter_error(real_pix_shift, pred_pix_shift, meters_in_pixel, relative=True)
    ppm = pixel_error_per_meter(
        a_meter=origin_meters[:-1],
        b_meter=origin_meters[1:],
        real_pix_shift=real_pix_shift[1:],
        pred_pix_shift=pred_pix_shift[1:],
    )
    return {
        "pixel_error": pixel_error,
        "absolute_meter_error": ame,
        "relative_meter_error": rme,
        "pixel_error_per_meter": np.concatenate([[0], ppm]),
    }


def qa_pd(gt: pd.DataFrame, pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    origin_meters = gt[["meter_center_x", "meter_center_y"]].to_numpy()
    real_pix_shift = gt[["pixel_shift_x", "pixel_shift_y"]].to_numpy()
    pred_pix_shift = pred[["pixel_shift_x", "pixel_shift_y"]].to_numpy()
    meters_in_pixel = meta[["meters_in_pixel_x", "meters_in_pixel_y"]].to_numpy()
    total = qa_np(origin_meters, real_pix_shift, pred_pix_shift, meters_in_pixel)
    df = pd.DataFrame(total)
    return df


def qa_csv(gt_path: str, pred_path: str, meta_path: str, dst_path: str) -> None:
    df = qa_pd(
        gt=pd.read_csv(gt_path),
        pred=pd.read_csv(pred_path),
        meta=pd.read_csv(meta_path),
    )
    df.to_csv(dst_path)
