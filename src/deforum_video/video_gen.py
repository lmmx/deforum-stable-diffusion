import os
import subprocess

__all__ = ["generate_video"]


def generate_video(args, anim_args, fps):
    image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.png")
    mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")
    print(f"{image_path} -> {mp4_path}")
    # make video
    cmd = [
        *"ffmpeg -y -vcodec png -r".split(),
        str(fps),
        "-start_number",
        str(0),
        "-i",
        image_path,
        "-frames:v",
        str(anim_args.max_frames),
        *"-c:v libx264 -vf".split(),
        f"fps={fps}",
        *"-pix_fmt yuv420p -crf 17 -preset veryfast".split(),
        mp4_path,
    ]
    print(" ".join(cmd))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
    # from base64 import b64encode
    # mp4 = open(mp4_path, "rb").read()
    # data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    # display.display(
    #     display.HTML(
    #         f'<video controls loop><source src="{data_url}" type="video/mp4"></video>'
    #     )
    # )
