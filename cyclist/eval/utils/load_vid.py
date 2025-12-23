
import av
import numpy as np


def read_video_pyav(vid_path, FPS=32, SAMPLED_FRAMES_PER_SEC=8):
    '''
    Decode the video with PyAV decoder and compute frame times.

    Args:
        vid_path (str): Path to the video file.
        FPS (int): Frame rate of the video.
        FRAMES_PER_SEC (int): Number of frames to sample per second.

    Returns:
        np.ndarray: Decoded frames of shape (num_frames, height, width, 3).
        float: Video duration in seconds.
        list: List of timestamps for sampled frames in seconds.
    '''
    try:
        container = av.open(vid_path)
    except av.error.InvalidDataError as e:
        print("Video path failed:", vid_path)
        print(e)
  

    # Get total frames and video duration
    total_frames = container.streams.video[0].frames
    stream = container.streams.video[0]
    

    
    time_base = stream.time_base  # Fraction representing time per frame
    if total_frames != 160:
        print("total frames",total_frames, vid_path) # TODO harcoded fix
        num_seconds = 160 / FPS
        vid_len = num_seconds * SAMPLED_FRAMES_PER_SEC # number of frames after sampling
    else:
        num_seconds = total_frames / FPS # video len in seconds
        vid_len = num_seconds * SAMPLED_FRAMES_PER_SEC # number of frames after sampling
        

    # Calculate indices for uniform sampling
    indices = np.arange(0, total_frames, total_frames / vid_len).astype(int)

    frames = []
    frame_times = []  # Store timestamps for sampled frames
    container.seek(0)
    
    start_index = indices[0]
    end_index = indices[-1]

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
            # Compute timestamp using pts and time_base
            frame_time = frame.pts * time_base if frame.pts is not None else 0
            frame_times.append(round(float(frame_time),2))  # Convert to seconds
    
    return np.stack(frames), num_seconds, frame_times

    
     