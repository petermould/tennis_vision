# Tennis-Vision

A computer vision pipeline for analyzing broadcast tennis footage — detecting and tracking players and the ball, calibrating real-world court geometry from a trained keypoint model, and deriving live match statistics (shot speed, player movement speed, rally structure) through homography-based coordinate mapping.

Built as part of an ongoing portfolio in sports analytics and computer vision, aimed at full-time roles in the field. A football/soccer adaptation of this same pipeline is in progress — see [Future work](#future-work).

## Demo

`output_videos/output_video_final.mp4`

*(Add the clip to this repo under `output_videos/` and it will render inline on GitHub. A short GIF preview works well here too.)*

## What it does

- **Player tracking** — YOLO11 + tracking, with automatic filtering to isolate the two actual players from officials, ball kids, and umpires (via average distance to all court keypoints, rather than nearest single point — see [Notable debugging decisions](#notable-debugging-decisions))
- **Ball tracking** — custom-trained YOLO model, with interpolation to smooth gaps in detection between frames
- **Court calibration** — a ResNet50 regression model trained to predict 14 court keypoints (corners, service lines, center marks) directly from a single frame
- **Homography mapping** — converts pixel positions (players, ball) into real-world court coordinates using known court dimensions and player height as calibration references
- **Mini-court visualization** — a live top-down court overlay showing player and ball positions in real time, including a fading ball trajectory trail
- **Match statistics** — shot speed, player movement speed, and running averages for both players, computed in m/s from real-world distances and elapsed time
- **On-court stat tags** — a fading "shot speed" tag appears beside whichever player just hit the ball; a persistent tag tracks current movement speed
- **Named player overlays** — bounding boxes labeled with player names rather than raw track IDs

## Pipeline
