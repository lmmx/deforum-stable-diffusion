import subprocess

from model_map import model_map

__all__ = ["wget", "download_model"]


def wget(url, outputdir):
    res = subprocess.run(
        ["wget", url, "-P", f"{outputdir}"], capture_output=True
    ).stdout.decode()
    print(res)
    return


def download_model(models_path, model_checkpoint):
    download_link = model_map[model_checkpoint]["link"][0]
    print(f"!wget -O {models_path}/{model_checkpoint} {download_link}")
    wget(download_link, models_path)
    return
