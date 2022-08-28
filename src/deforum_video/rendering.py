import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from animation import get_inbetweens, make_xform_2d, parse_key_frames
from modelling import (
    add_noise,
    generate,
    maintain_colors,
    sample_from_cv2,
    sample_to_cv2,
)
from prompts import animation_prompts, prompts
from seeding import next_seed
from video_gen import generate_video

__all__ = [
    "device",
    "render_image_batch",
    "render_animation",
    "render_input_video",
    "run_render",
]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def render_image_batch(args, model):
    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    if args.save_settings or args.save_samples:
        print(f"Saving to {os.path.join(args.outdir, args.timestring)}_*")
    # save settings for the batch
    if args.save_settings:
        filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
        with open(filename, "w+", encoding="utf-8") as f:
            json.dump(dict(args.__dict__), f, ensure_ascii=False, indent=4)
    index = 0
    for batch_index in range(args.n_batch):
        print(f"Batch {batch_index+1} of {args.n_batch}")
        for prompt in prompts:
            args.prompt = prompt
            results = generate(args=args, model=model)
            for image in results:
                if args.save_samples:
                    filename = f"{args.timestring}_{index:05}_{args.seed}.png"
                    image.save(os.path.join(args.outdir, filename))
                # if args.display_samples:
                #     display.display(image)
                index += 1
            args.seed = next_seed(args)


def render_animation(args, anim_args, model):
    if anim_args.key_frames:
        angle_series = get_inbetweens(parse_key_frames(anim_args.angle))
        zoom_series = get_inbetweens(parse_key_frames(anim_args.zoom))
        translation_x_series = get_inbetweens(parse_key_frames(anim_args.translation_x))
        translation_y_series = get_inbetweens(parse_key_frames(anim_args.translation_y))
    # animations use key framed prompts
    args.prompts = animation_prompts
    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving animation frames to {args.outdir}")
    # save settings for the batch
    settings_filename = os.path.join(args.outdir, f"{args.timestring}_settings.txt")
    with open(settings_filename, "w+", encoding="utf-8") as f:
        s = {**dict(args.__dict__), **dict(anim_args.__dict__)}
        json.dump(s, f, ensure_ascii=False, indent=4)
    # expand prompts out to per-frame
    prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames)])
    for i, prompt in animation_prompts.items():
        prompt_series[i] = prompt
    prompt_series = prompt_series.ffill().bfill()
    # check for video inits
    using_vid_init = anim_args.animation_mode == "Video Input"
    args.n_samples = 1
    prev_sample = None
    color_match_sample = None
    for frame_idx in range(anim_args.max_frames):
        print(f"Rendering animation frame {frame_idx} of {anim_args.max_frames}")
        # apply transforms to previous frame
        if prev_sample is not None:
            if anim_args.key_frames:
                angle = angle_series[frame_idx]
                zoom = zoom_series[frame_idx]
                translation_x = translation_x_series[frame_idx]
                translation_y = translation_y_series[frame_idx]
                print(
                    f"angle: {angle}",
                    f"zoom: {zoom}",
                    f"translation_x: {translation_x}",
                    f"translation_y: {translation_y}",
                )
            xform = make_xform_2d(
                args.W, args.H, translation_x, translation_y, angle, zoom
            )
            # transform previous frame
            prev_img = sample_to_cv2(prev_sample)
            prev_img = cv2.warpPerspective(
                prev_img,
                xform,
                (prev_img.shape[1], prev_img.shape[0]),
                borderMode=cv2.BORDER_WRAP
                if anim_args.border == "wrap"
                else cv2.BORDER_REPLICATE,
            )
            # apply color matching
            if anim_args.color_coherence == "MatchFrame0":
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()
                else:
                    prev_img = maintain_colors(
                        prev_img, color_match_sample, (frame_idx % 2) == 0
                    )
            # apply frame noising
            noised_sample = add_noise(
                sample_from_cv2(prev_img), anim_args.previous_frame_noise
            )
            # use transformed previous frame as init for current
            args.use_init = True
            args.init_sample = noised_sample.half().to(device)
            args.strength = max(0.0, min(1.0, anim_args.previous_frame_strength))
        # grab prompt for current frame
        args.prompt = prompt_series[frame_idx]
        print(f"{args.prompt} {args.seed}")
        # grab init image for current frame
        if using_vid_init:
            init_frame = os.path.join(
                args.outdir, "inputframes", f"{frame_idx+1:04}.jpg"
            )
            print(f"Using video init frame {init_frame}")
            args.init_image = init_frame
        # sample the diffusion model
        results = generate(
            args=args, model=model, return_latent=False, return_sample=True
        )
        sample, image = results[0], results[1]
        filename = f"{args.timestring}_{frame_idx:05}.png"
        image.save(os.path.join(args.outdir, filename))
        if not using_vid_init:
            prev_sample = sample
        # display.clear_output(wait=True)
        # display.display(image)
        args.seed = next_seed(args)


def render_input_video(args, anim_args, model):
    # create a folder for the video input frames to live in
    video_in_frame_path = os.path.join(args.outdir, "inputframes")
    os.makedirs(os.path.join(args.outdir, video_in_frame_path), exist_ok=True)
    # save the video frames from input video
    print(
        f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {video_in_frame_path}..."
    )
    try:
        for f in Path(video_in_frame_path).glob("*.jpg"):
            f.unlink()
    except BaseException:
        pass
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            f"{anim_args.video_init_path}",
            "-vf",
            f"select=not(mod(n\,{anim_args.extract_nth_frame}))",
            *"-vsync vfr -q:v 2 -loglevel error -stats".split(),
            os.path.join(video_in_frame_path, "%04d.jpg"),
        ],
        capture_output=True,
    ).stdout.decode()
    # determine max frames from length of input frames
    anim_args.max_frames = len([f for f in Path(video_in_frame_path).glob("*.jpg")])
    args.use_init = True
    print(
        f"Loading {anim_args.max_frames} input frames from {video_in_frame_path} and saving video frames to {args.outdir}"
    )
    render_animation(args=args, anim_args=anim_args, model=model)


def run_render(args, anim_args, model, skip_video, fps):
    if anim_args.animation_mode == "2D":
        render_animation(args=args, anim_args=anim_args, model=model)
    elif anim_args.animation_mode == "Video Input":
        render_input_video(args=args, anim_args=anim_args, model=model)
    else:
        render_image_batch(args=args, model=model)
    if skip_video:
        print("Skipping video creation, uncheck skip_video if you want to run it")
    else:
        generate_video(args=args, anim_args=anim_args, fps=fps)
