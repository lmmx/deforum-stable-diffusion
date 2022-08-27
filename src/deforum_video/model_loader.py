import torch
from ldm.util import instantiate_from_config

# this should already see sys.path.append('./k-diffusion')


__all__ = ["load_model_from_config"]


def load_model_from_config(config, ckpt, verbose=False, device="cuda"):
    map_location = "cuda"
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=map_location)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model = model.half().to(device)
    model.eval()
    return model
