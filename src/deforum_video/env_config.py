import subprocess

print("...setting up environment")
print_subprocess = False
all_process = [
    [
        "pip",
        "install",
        "torch==1.11.0+cu113",
        "torchvision==0.12.0+cu113",
        "torchaudio==0.11.0",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu113",
    ],
    [
        "pip",
        "install",
        "omegaconf==2.1.1",
        "einops==0.3.0",
        "pytorch-lightning==1.4.2",
        "torchmetrics==0.6.0",
        "torchtext==0.2.3",
        "transformers==4.19.2",
        "kornia==0.6",
    ],
    ["git", "clone", "https://github.com/deforum/stable-diffusion"],
    [
        "pip",
        "install",
        "-e",
        "git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers",
    ],
    ["pip", "install", "-e", "git+https://github.com/openai/CLIP.git@main#egg=clip"],
    [
        "pip",
        "install",
        "accelerate",
        "ftfy",
        "jsonmerge",
        "resize-right",
        "torchdiffeq",
    ],
]
for process in all_process:
    running = subprocess.run(process, stdout=subprocess.PIPE).stdout.decode("utf-8")
    if print_subprocess:
        print(running)

print(
    subprocess.run(
        ["git", "clone", "https://github.com/deforum/k-diffusion/"],
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")
)
with open("k-diffusion/k_diffusion/__init__.py", "w") as f:
    f.write("")
