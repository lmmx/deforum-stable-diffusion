import shutil

from omegaconf import OmegaConf

from anim_config import anim_args, process_anim_args
from downloading import download_model
from env_config import report_env
from general_config import general_args as args
from general_config import (
    local_deforum_sd,
    make_models_and_output_dirs,
    models_path,
    process_args,
)
from model_hash import check_model_hash
from model_loader import load_model_from_config
from rendering import device, run_render

__all__ = ["main"]


def main(
    skip_video=True,
    fps=12,
    check_sha256=True,
    model_config="v1-inference.yaml",
    model_checkpoint="sd-v1-4.ckpt",
    custom_config_path="",
    custom_checkpoint_path="",
):
    report_env()
    print("Local Path Variables:\n")
    make_models_and_output_dirs()
    sd_config_dir = local_deforum_sd / "configs" / "stable-diffusion"
    # config path
    if (model_config_path := (models_path / model_config)).exists():
        print(f"{model_config_path=} exists")
    else:
        print(f"cp {sd_config_dir}/{model_config} $models_path/.")
        shutil.copy(f"{sd_config_dir}/{model_config}", models_path)
    # checkpoint path or download
    if (model_ckpt_path := (models_path / model_checkpoint)).exists():
        print(f"{model_ckpt_path=} exists")
    else:
        v14url = "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original"
        print(
            f"Download model checkpoint and place in {model_ckpt_path=}.\n"
            f" E.g. for v1.4 go accept T&Cs at {v14url}"
        )
        # download_model(models_path=models_path, model_checkpoint=model_checkpoint)
        raise ValueError("STOP!")
    if check_sha256:
        check_model_hash(models_path, model_checkpoint)
    config = custom_config_path if model_config == "custom" else model_config_path
    ckpt = custom_checkpoint_path if model_checkpoint == "custom" else model_ckpt_path
    print(f"config: {config}")
    print(f"ckpt: {ckpt}")
    local_config = OmegaConf.load(f"{config}")
    model = load_model_from_config(local_config, f"{ckpt}")
    model = model.to(device)
    process_anim_args(anim_args=anim_args)
    process_args(args=args, anim_args=anim_args)
    run_render(
        args=args, anim_args=anim_args, model=model, skip_video=skip_video, fps=fps
    )
    return


if __name__ == "__main__":
    main()
