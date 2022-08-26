from __future__ import annotations

import re

import cv2
import numpy as np
import pandas as pd

from anim_config import anim_args

__all__ = ["make_xform_2d", "parse_key_frames", "get_inbetweens"]


def make_xform_2d(width, height, translation_x, translation_y, angle, scale):
    center = (height // 2, width // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    trans_mat = np.vstack([trans_mat, [0, 0, 1]])
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    return np.matmul(rot_mat, trans_mat)


def parse_key_frames(string, prompt_parser=None) -> dict:
    pattern = r"((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])"
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()["frame"])
        param = match_object.groupdict()["param"]
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param
    if frames == {} and len(string) != 0:
        raise RuntimeError("Key Frame string not correctly formatted")
    return frames


def get_inbetweens(key_frames, integer=False):
    key_frame_series = pd.Series([np.nan for a in range(anim_args.max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    interp_method = anim_args.interp_spline
    if interp_method == "Cubic" and len(key_frames.items()) <= 3:
        interp_method = "Quadratic"
    if interp_method == "Quadratic" and len(key_frames.items()) <= 2:
        interp_method = "Linear"

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[anim_args.max_frames - 1] = key_frame_series[
        key_frame_series.last_valid_index()
    ]
    key_frame_series = key_frame_series.interpolate(
        method=interp_method.lower(), limit_direction="both"
    )
    return key_frame_series.astype(int) if integer else key_frame_series
