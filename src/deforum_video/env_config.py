import subprocess

__all__ = ["report_env", "setup_env"]


def report_env(setup_environment=False):
    sub_p_res = subprocess.run(
        """
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        """.split(),
        capture_output=True,
    ).stdout.decode()
    print(sub_p_res)
    if setup_environment:
        setup_env()


def setup_env(print_subprocess=False):
    """
    For reference only: incompatible with this library after rewriting from the
    notebook. Follow the guide in `CONDA_SETUP.md`
    """
    print("...setting up environment")
    all_process = [
        [
            *"pip install".split(),
            "torch==1.11.0+cu113",
            "torchvision==0.12.0+cu113",
            "torchaudio==0.11.0",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu113",
        ],
        [
            *"pip install".split(),
            "omegaconf==2.1.1",
            "einops==0.3.0",
            "pytorch-lightning==1.4.2",
            "torchmetrics==0.6.0",
            "torchtext==0.2.3",
            "transformers==4.19.2",
            "kornia==0.6",
        ],
        "git clone https://github.com/deforum/stable-diffusion".split(),
        [
            *"pip install -e".split(),
            "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers",
        ],
        [
            *"pip install -e".split(),
            "git+https://github.com/openai/CLIP.git@main#egg=clip",
        ],
        [
            *"pip install".split(),
            *"accelerate ftfy jsonmerge resize-right torchdiffeq".split(),
        ],
    ]
    for process in all_process:
        running = subprocess.run(process, capture_output=True).stdout.decode()
        if print_subprocess:
            print(running)

    print(
        subprocess.run(
            "git clone https://github.com/deforum/k-diffusion/".split(),
            capture_output=True,
        ).stdout.decode()
    )
    with open("k-diffusion/k_diffusion/__init__.py", "w") as f:
        f.write("")
