# deforum-stable-diffusion

Refactor of the Deforum Stable Diffusion notebook (featuring video_init) https://colab.research.google.com/github/deforum/stable-diffusion/blob/main/Deforum_Stable_Diffusion.ipynb

- v0.1 announcement: https://twitter.com/deforum_art/status/1562999935414333441

## Layout

The notebook has been split into the following parts:

- `deforum_video.py` is the main module (everything else gets imported via that if used directly)
  - Reports on the GPU using `nvidia-smi`
- `general_config.py`:
  - sets `models_path` and `output_path` and creates them if they don't exist (they're no longer at
    `/content/models` and `/content/output` but under the caller's current working directory `./`)
  - also sets `DeforumArgs` and the namespace (`general_args`) which gets imported to the main
    `deforum_video` module as the name `args` (like in the original notebook).
- `env_config.py` 

## Installation

See `CONDA_SETUP.md` file indicating prerequisites to run and the reasoning behind installing them this way

```sh
git clone https://github.com/lmmx/deforum-stable-diffusion
cd deforum-stable-diffusion
conda create -n deforum
conda activate deforum
conda install -y pytorch torchvision torchaudio cudatoolkit=11.6 pytorch-lightning -c pytorch -c conda-forge
conda install ffmpeg
pip install accelerate einops ftfy jsonmerge kornia omegaconf opencv-python pandas resize-right torchdiffeq torchmetrics torchtext==0.2.3 transformers
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/deforum/stable-diffusion@main#egg=latent-diffusion
pip install -e git+https://github.com/deforum/k-diffusion@master#egg=k-diffusion
```

- **Note**: somehow I ended up with a broken ffmpeg, but after `conda uninstall ffmpeg`
  and `conda install ffmpeg` again  it fixed itself

To download the weights, you need to accept their T&Cs, e.g. for v1.4 go here

- https://huggingface.co/CompVis/stable-diffusion-v-1-4-original

(This can be made less awkward by uncommenting the `download_model` function)

## Usage

For now run with:

```sh
python ./src/deforum_video/deforum_video.py
```

and edit settings in `./src/deforum_video/anim_config.py`.

- If you choose `animation_mode="2D"` you get a video from the prompt(s)
- If you choose `animation_mode="Video Input"` you get a video from applying the prompts to the
  video at `./content/input/video_in.mp4` (this path is customisable as the `video_init_path`).
- Change the output length with `max_frames`, etc. etc.

I will improve this and make it a proper package CLI with entrypoints etc. soon !
