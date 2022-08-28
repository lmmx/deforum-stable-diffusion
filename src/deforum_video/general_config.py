import os
from types import SimpleNamespace

from modelling import get_output_folder

__all__ = ["make_models_and_output_dirs", "resize_side", "process_args", "DeforumArgs", "general_args"]


batch_name = "StableFun"
models_path = "./content/models"
output_path = "./content/output"


def make_models_and_output_dirs():
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    print(f"{models_path=}\n{output_path=}")


def resize_side(side):
    """Resize to integer multiple of 64"""
    return side - side % 64


def process_args(args, anim_args):
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
    return


DeforumArgs = dict(
    # Save & Display Settings
    batch_name=batch_name,
    outdir=get_output_folder(output_path, batch_name),
    save_grid=False,
    save_settings=True,
    save_samples=True,
    display_samples=False,
    # Image Settings
    n_samples=1,
    W=resize_side(512),
    H=resize_side(512),
    # Init Settings
    use_init=False,
    strength=0.5,
    init_image="https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg",
    # Sampling Settings
    seed=-1,
    sampler="klms",  # ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    steps=100,
    scale=7,
    ddim_eta=0.0,
    dynamic_threshold=None,
    static_threshold=None,
    # Batch Settings
    n_batch=1,
    seed_behavior="iter",  # @param ["iter","fixed","random"]
    precision="autocast",
    fixed_code=True,
    C=4,
    f=8,
    prompt="",
    timestring="",
    init_latent=None,
    init_sample=None,
)

general_args = SimpleNamespace(**DeforumArgs)
