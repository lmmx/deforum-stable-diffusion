from types import SimpleNamespace

__all__ = ["process_anim_args", "DeforumAnimArgs", "anim_args"]


def process_anim_args(anim_args):
    if anim_args.animation_mode == "None":
        anim_args.max_frames = 1


DeforumAnimArgs = dict(
    # Animation
    animation_mode="None",  # @param ['None', '2D', 'Video Input'] {type:'string'}
    max_frames=1000,  # @param {type:"number"}
    border="wrap",  # @param ['wrap', 'replicate'] {type:'string'}
    # Motion Parameters
    key_frames=True,  # @param {type:"boolean"}
    interp_spline="Linear",  # Do not change, currently will not look good. param ['Linear','Quadratic','Cubic']{type:"string"}
    angle="0:(0)",  # @param {type:"string"}
    zoom="0: (1.04)",  # @param {type:"string"}
    translation_x="0: (0)",  # @param {type:"string"}
    translation_y="0: (0)",  # @param {type:"string"}
    # Coherence
    color_coherence="MatchFrame0",  # @param ['None', 'MatchFrame0'] {type:'string'}
    previous_frame_noise=0.02,  # @param {type:"number"}
    previous_frame_strength=0.65,  # @param {type:"number"}
    # Video Input
    video_init_path="/content/video_in.mp4",  # @param {type:"string"}
    extract_nth_frame=1,  # @param {type:"number"}
)


anim_args = SimpleNamespace(**DeforumAnimArgs)
