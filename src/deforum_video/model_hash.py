import hashlib
from pathlib import Path

from model_map import model_map


def check_model_hash(models_path, model_checkpoint):
    print("\n...checking sha256")
    with open(Path(models_path) / model_checkpoint, "rb") as f:
        bytes = f.read()
        hash = hashlib.sha256(bytes).hexdigest()
        del bytes
    if model_map[model_checkpoint]["sha256"] == hash:
        print("hash is correct\n")
    else:
        print("hash in not correct\n")
