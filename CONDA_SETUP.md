My original diffusers conda environment setup was:

```sh
conda create -n diffusers
conda activate diffusers
conda install -y pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install diffusers transformers piexif
```

The module `env_config.py` specifies the following:

```sh
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install omegaconf==2.1.1 einops==0.3.0 pytorch-lightning==1.4.2 torchmetrics==0.6.0 torchtext==0.2.3 transformers==4.19.2 kornia==0.6
git clone https://github.com/deforum/stable-diffusion
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install accelerate ftfy jsonmerge resize-right torchdiffeq
git clone https://github.com/deforum/k-diffusion/
echo > k-diffusion/k_diffusion/__init__.py
```

...however on your own system (as opposed to in a Colab notebook) I do **not** recommend this.

- For one, it uses pip not conda (so can't take advantage of mkl accelerated numpy etc).
- Secondly, it uses CUDA 11.3 specifically, whereas 11.6 is available.

I looked into the effect of the final `echo` (that blanks the init file) and came to the conclusion
its purpose is to avoid (perhaps slow) loading of: `augmentation`, `config`, `evaluation`, `gns`, `layers`, `models` modules,
(perhaps so that it avoids needing to incorporate the package-level requirement `jsonmerge` which is
imported in `config`, or `clean-fid` in `evaluation`...)

In light of this, it'd still be simpler I think to just git clone it, overwrite the `__init__`
module, and then install it in editable mode. (I don't imagine those dependencies are troublesome)

If you do this, you get CLIP too, as the last line of the `k-diffusion` package's requirements is
`git+https://github.com/openai/CLIP`.

Another thing to note is that `pip install -e git+GIT_URL#egg=foo` is the same as `git clone GIT_URL src/foo; pip install -e src/foo`.

`deforum/stable-diffusion` won't let you do this as the metadata has the name `latent-diffusion`, so
you must pass this as the `#egg` name in the URL param or it'll error out cautiously.
The internal name is `ldm` (as in, the code calls `import from ldm.models... import ...`)

With this in mind, to recreate an environment:

```sh
git clone https://github.com/lmmx/deforum-stable-diffusion
cd deforum-stable-diffusion
conda create -n deforum
conda activate deforum
conda install -y pytorch torchvision torchaudio torchtext cudatoolkit=11.6 pytorch-lightning -c pytorch -c conda-forge
pip install accelerate einops ftfy jsonmerge kornia omegaconf opencv-python resize-right torchdiffeq torchmetrics transformers
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/deforum/stable-diffusion@main#egg=latent-diffusion
pip install -e git+https://github.com/deforum/k-diffusion@master#egg=k-diffusion
```

- Note that it's important to `cd` into the package after you clone it so that `pip install -e git+https://...`
  will put packages into `src/` of this repo, where it can detect the config files (not accessible via Python namespaces).

Having done this, I've turned off the `subprocess` approach to installation in the code
(`report_env()` takes an arg `setup_environment` that defaults to False).
