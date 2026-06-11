import av
import numpy as np


def read_video_pyav(vid_path, FPS=32, SAMPLED_FRAMES_PER_SEC=8):
    '''
    Decode the video with PyAV and return uniformly sampled frames with timestamps.

    Args:
        vid_path (str): Path to the video file.
        FPS (int): Native frame rate of the video.
        SAMPLED_FRAMES_PER_SEC (int): Number of frames to sample per second.

    Returns:
        np.ndarray: Decoded frames of shape (num_frames, height, width, 3).
        float: Video duration in seconds.
        list: Timestamps of sampled frames in seconds.
    '''
    try:
        container = av.open(vid_path)
    except av.error.InvalidDataError as e:
        print("Video path failed:", vid_path)
        print(e)
        return None, None, None

    stream = container.streams.video[0]
    total_frames = stream.frames
    time_base = stream.time_base

    # 160 is the expected frame count for a 5-second video at 32 fps
    if total_frames != 160:
        print(f"Warning: {vid_path} has {total_frames} frames, expected 160.")
        num_seconds = 160 / FPS
    else:
        num_seconds = total_frames / FPS

    vid_len = num_seconds * SAMPLED_FRAMES_PER_SEC
    indices = np.arange(0, total_frames, total_frames / vid_len).astype(int)

    frames = []
    frame_times = []
    container.seek(0)
    end_index = indices[-1]

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
            frame_time = frame.pts * time_base if frame.pts is not None else 0
            frame_times.append(round(float(frame_time), 2))

    return np.stack(frames), num_seconds, frame_times
