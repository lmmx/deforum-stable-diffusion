import os
import random
import shutil
import subprocess
import time

import torch
from omegaconf import OmegaConf

from anim_config import anim_args
from env_config import setup_env
from general_config import general_args as args
from model_hash import check_model_hash
from model_loader import load_model_from_config
from rendering import render_animation, render_image_batch, render_input_video
from video_gen import generate_video

sub_p_res = subprocess.run(
    "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader".split(),
    capture_output=True,
).stdout.decode()
print(sub_p_res)

print("Local Path Variables:\n")

setup_environment = False

if setup_environment:
    setup_env()

model_config = "v1-inference.yaml"
model_checkpoint = "sd-v1-4.ckpt"
custom_config_path = ""
custom_checkpoint_path = ""
check_sha256 = True

# config path
if os.path.exists(model_config_path := (models_path + "/" + model_config)):
    print(f"{model_config_path=} exists")
else:
    print(
        "cp ./stable-diffusion/configs/stable-diffusion/v1-inference.yaml $models_path/."
    )
    shutil.copy(
        "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml", models_path
    )

# checkpoint path or download
if os.path.exists(models_path + "/" + model_checkpoint):
    print(f"{models_path+'/'+model_checkpoint} exists")
else:
    print(f"download model checkpoint and place in {models_path+'/'+model_checkpoint}")
    # download_model(models_path=models_path, model_checkpoint=model_checkpoint)

if check_sha256:
    check_model_hash(models_path, model_checkpoint)

if model_config == "custom":
    config = custom_config_path
else:
    config = models_path + "/" + model_config

if model_checkpoint == "custom":
    ckpt = custom_checkpoint_path
else:
    ckpt = models_path + "/" + model_checkpoint

print(f"config: {config}")
print(f"ckpt: {ckpt}")

load_on_run_all = True

if load_on_run_all:
    local_config = OmegaConf.load(f"{config}")
    model = load_model_from_config(local_config, f"{ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

if anim_args.animation_mode == "None":
    anim_args.max_frames = 1

args.timestring = time.strftime("%Y%m%d%H%M%S")
args.strength = max(0.0, min(1.0, args.strength))

if args.seed == -1:
    args.seed = random.randint(0, 2 ** 32)
if anim_args.animation_mode == "Video Input":
    args.use_init = True
if not args.use_init:
    args.init_image = None
    args.strength = 0
if args.sampler == "plms" and (args.use_init or anim_args.animation_mode != "None"):
    print("Init images aren't supported with PLMS yet, switching to KLMS")
    args.sampler = "klms"
if args.sampler != "ddim":
    args.ddim_eta = 0

if anim_args.animation_mode == "2D":
    render_animation(args=args, anim_args=anim_args)
elif anim_args.animation_mode == "Video Input":
    render_input_video(args, anim_args)
else:
    render_image_batch(args)

skip_video_for_run_all = True
fps = 12

if skip_video_for_run_all:
    print(
        "Skipping video creation, uncheck skip_video_for_run_all if you want to run it"
    )
else:
    generate_video(args=args, anim_args=anim_args, fps=fps)
