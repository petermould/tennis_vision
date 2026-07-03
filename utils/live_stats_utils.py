import cv2


def draw_player_speed_tag(frame, bbox, speed_mps):
    """Small persistent pill under the player's bbox showing current movement speed."""
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    y = int(y2) + 10

    text = f"{speed_mps:.1f} m/s"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (cx - tw // 2 - 8, y), (cx + tw // 2 + 8, y + th + 10), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, text, (cx - tw // 2, y + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return frame


def draw_shot_speed_tag(frame, bbox, speed_mps, frames_since_shot, fade_duration=40):
    """Fading tag beside the player, shown briefly right after they hit the ball."""
    if frames_since_shot > fade_duration or frames_since_shot < 0:
        return frame

    alpha = max(0.0, 1.0 - (frames_since_shot / fade_duration))
    x1, y1, x2, y2 = bbox
    tag_x, tag_y = int(x2) + 15, int(y1) + 30

    overlay = frame.copy()
    text = f"{speed_mps:.1f} m/s"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    box_w = tw + 20
    cv2.rectangle(overlay, (tag_x, tag_y - th - 14), (tag_x + box_w, tag_y + 6), (25, 197, 240), -1)
    cv2.putText(overlay, "shot speed", (tag_x + 10, tag_y - th - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 40, 10), 1, cv2.LINE_AA)
    cv2.putText(overlay, text, (tag_x + 10, tag_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 30, 60), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_live_stat_tags(video_frames, player_detections, shot_events, player_stats_data_df,
                         id_to_label, fps=60, fade_seconds=1.5):
    """
    video_frames: list of frames to draw on (already drawn with boxes/names etc.)
    player_detections: list of {track_id: bbox} per frame, real pixel coordinates
    shot_events: list of {'frame': int, 'player_id': real_track_id, 'speed': float}
    player_stats_data_df: forward-filled per-frame DataFrame with player_{1,2}_last_player_speed
    id_to_label: dict mapping real track_id -> 1 or 2 (matches player_stats_data_df columns)
    """
    fade_duration = int(fps * fade_seconds)

    # sort shot events by frame for lookup
    shot_events_sorted = sorted(shot_events, key=lambda e: e['frame'])

    for frame_num, frame in enumerate(video_frames):
        player_dict = player_detections[frame_num]
        stats_row = player_stats_data_df.iloc[frame_num] if frame_num < len(player_stats_data_df) else None

        for track_id, bbox in player_dict.items():
            label = id_to_label.get(track_id)
            if label is None:
                continue

            # player speed tag (always shown, from forward-filled stats)
            if stats_row is not None:
                speed_col = f'player_{label}_last_player_speed'
                if speed_col in stats_row and stats_row[speed_col] > 0:
                    frame = draw_player_speed_tag(frame, bbox, stats_row[speed_col])

            # find most recent shot by this player at or before this frame
            last_event = None
            for event in shot_events_sorted:
                if event['frame'] > frame_num:
                    break
                if event['player_id'] == track_id:
                    last_event = event

            if last_event is not None:
                frames_since_shot = frame_num - last_event['frame']
                frame = draw_shot_speed_tag(frame, bbox, last_event['speed'], frames_since_shot, fade_duration)

        video_frames[frame_num] = frame

    return video_frames