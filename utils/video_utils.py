import cv2

def read_video(video_path):
    """
    Reads a video file and returns a list of frames.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: A list of frames (numpy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, 50, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    
    