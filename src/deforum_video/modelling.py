import os
import sys
import time
from contextlib import nullcontext

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from einops import rearrange, repeat
from k_diffusion import sampling
from k_diffusion.external import CompVisDenoiser
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from PIL import Image
from pytorch_lightning import seed_everything
from skimage.exposure import match_histograms
from torch import autocast

# The following lines were in the notebook to allow k_diffusion and ldm to be imported
# from without installing them (unclear why they weren't installed in editable mode)
# sys.path.append("./src/taming-transformers")
# sys.path.append("./src/clip")
# sys.path.append("./stable-diffusion/")
# sys.path.append("./k-diffusion")


__all__ = [
    "CFGDenoiser",
    "add_noise",
    "get_output_folder",
    "load_img",
    "maintain_colors",
    "make_callback",
    "generate",
    "sample_from_cv2",
    "sample_to_cv2",
]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


def add_noise(sample: torch.Tensor, noise_amt: float):
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt


def get_output_folder(output_path, batch_folder=None):
    yearMonth = time.strftime("%Y-%m/")
    out_path = output_path + "/" + yearMonth
    if batch_folder != "":
        out_path += batch_folder
        if out_path[-1] != "/":
            out_path += "/"
    os.makedirs(out_path, exist_ok=True)
    return out_path


def load_img(path, shape):
    if path.startswith("http://") or path.startswith("https://"):
        image = Image.open(requests.get(path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(path).convert("RGB")

    image = image.resize(shape, resample=Image.LANCZOS)
    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def maintain_colors(prev_img, color_match_sample, hsv=False):
    if hsv:
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else:
        return match_histograms(prev_img, color_match_sample, multichannel=True)


def make_callback(sampler, dynamic_threshold=None, static_threshold=None):
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image after each step
    def dynamic_thresholding_(img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1, img.ndim)))
        s = np.max(np.append(s, 1.0))
        torch.clamp_(img, -1 * s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback(args_dict):
        if static_threshold is not None:
            torch.clamp_(args_dict["x"], -1 * static_threshold, static_threshold)
        if dynamic_threshold is not None:
            dynamic_thresholding_(args_dict["x"], dynamic_threshold)

    # Function that is called on the image (img) and step (i) at each step
    def img_callback(img, i):
        # Thresholding functions
        if dynamic_threshold is not None:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(img, -1 * static_threshold, static_threshold)

    if sampler in ["plms", "ddim"]:
        # Callback function formated for compvis latent diffusion samplers
        callback = img_callback
    else:
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback

    return callback


def generate(args, return_latent=False, return_sample=False):
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    if args.sampler == "plms":
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    model_wrap = CompVisDenoiser(model)
    batch_size = args.n_samples
    prompt = args.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    init_latent = None
    if args.init_latent is not None:
        init_latent = args.init_latent
    elif args.init_sample is not None:
        init_latent = model.get_first_stage_encoding(
            model.encode_first_stage(args.init_sample)
        )
    elif args.init_image is not None and args.init_image != "":
        init_image = load_img(args.init_image, shape=(args.W, args.H)).to(device)
        init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
        init_latent = model.get_first_stage_encoding(
            model.encode_first_stage(init_image)
        )  # move to latent space

    sampler.make_schedule(
        ddim_num_steps=args.steps, ddim_eta=args.ddim_eta, verbose=False
    )

    t_enc = int((1.0 - args.strength) * args.steps)

    start_code = None
    if args.fixed_code and init_latent is not None:
        start_code = torch.randn(
            [args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device
        )

    callback = make_callback(
        sampler=args.sampler,
        dynamic_threshold=args.dynamic_threshold,
        static_threshold=args.static_threshold,
    )

    results = []
    precision_scope = autocast if args.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in range(args.n_samples):
                    for prompts in data:
                        uc = None
                        if args.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        if args.sampler in [
                            "klms",
                            "dpm2",
                            "dpm2_ancestral",
                            "heun",
                            "euler",
                            "euler_ancestral",
                        ]:
                            shape = [args.C, args.H // args.f, args.W // args.f]
                            sigmas = model_wrap.get_sigmas(args.steps)
                            if args.use_init:
                                sigmas = sigmas[len(sigmas) - t_enc - 1 :]
                                x = (
                                    init_latent
                                    + torch.randn(
                                        [args.n_samples, *shape], device=device
                                    )
                                    * sigmas[0]
                                )
                            else:
                                x = (
                                    torch.randn([args.n_samples, *shape], device=device)
                                    * sigmas[0]
                                )
                            model_wrap_cfg = CFGDenoiser(model_wrap)
                            extra_args = {
                                "cond": c,
                                "uncond": uc,
                                "cond_scale": args.scale,
                            }
                            if args.sampler == "klms":
                                samples = sampling.sample_lms(
                                    model_wrap_cfg,
                                    x,
                                    sigmas,
                                    extra_args=extra_args,
                                    disable=False,
                                    callback=callback,
                                )
                            elif args.sampler == "dpm2":
                                samples = sampling.sample_dpm_2(
                                    model_wrap_cfg,
                                    x,
                                    sigmas,
                                    extra_args=extra_args,
                                    disable=False,
                                    callback=callback,
                                )
                            elif args.sampler == "dpm2_ancestral":
                                samples = sampling.sample_dpm_2_ancestral(
                                    model_wrap_cfg,
                                    x,
                                    sigmas,
                                    extra_args=extra_args,
                                    disable=False,
                                    callback=callback,
                                )
                            elif args.sampler == "heun":
                                samples = sampling.sample_heun(
                                    model_wrap_cfg,
                                    x,
                                    sigmas,
                                    extra_args=extra_args,
                                    disable=False,
                                    callback=callback,
                                )
                            elif args.sampler == "euler":
                                samples = sampling.sample_euler(
                                    model_wrap_cfg,
                                    x,
                                    sigmas,
                                    extra_args=extra_args,
                                    disable=False,
                                    callback=callback,
                                )
                            elif args.sampler == "euler_ancestral":
                                samples = sampling.sample_euler_ancestral(
                                    model_wrap_cfg,
                                    x,
                                    sigmas,
                                    extra_args=extra_args,
                                    disable=False,
                                    callback=callback,
                                )
                        else:

                            if init_latent is not None:
                                z_enc = sampler.stochastic_encode(
                                    init_latent,
                                    torch.tensor([t_enc] * batch_size).to(device),
                                )
                                samples = sampler.decode(
                                    z_enc,
                                    c,
                                    t_enc,
                                    unconditional_guidance_scale=args.scale,
                                    unconditional_conditioning=uc,
                                )
                            else:
                                if args.sampler == "plms" or args.sampler == "ddim":
                                    shape = [args.C, args.H // args.f, args.W // args.f]
                                    samples, _ = sampler.sample(
                                        S=args.steps,
                                        conditioning=c,
                                        batch_size=args.n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=args.scale,
                                        unconditional_conditioning=uc,
                                        eta=args.ddim_eta,
                                        x_T=start_code,
                                        img_callback=callback,
                                    )

                        if return_latent:
                            results.append(samples.clone())

                        x_samples = model.decode_first_stage(samples)
                        if return_sample:
                            results.append(x_samples.clone())

                        x_samples = torch.clamp(
                            (x_samples + 1.0) / 2.0, min=0.0, max=1.0
                        )

                        for x_sample in x_samples:
                            x_sample = 255.0 * rearrange(
                                x_sample.cpu().numpy(), "c h w -> h w c"
                            )
                            image = Image.fromarray(x_sample.astype(np.uint8))
                            results.append(image)
    return results


def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample


def sample_to_cv2(sample: torch.Tensor) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(
        np.float32
    )
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255).astype(np.uint8)
    return sample_int8
