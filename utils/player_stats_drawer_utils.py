import numpy as np
import cv2


def draw_player_stats(output_video_frames, player_stats, player_names={1: "R. Federer", 2: "N. Djokovic"}):
    width = 360
    height = 250

    for index, row in player_stats.iterrows():
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']
        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']

        frame = output_video_frames[index]

        # position: left, vertically centered
        start_x = 40
        start_y = (frame.shape[0] // 2) - (height // 2)
        end_x = start_x + width
        end_y = start_y + height

        # dark panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # header strip, slightly lighter than the body
        header_h = 42
        header_overlay = frame.copy()
        cv2.rectangle(header_overlay, (start_x, start_y), (end_x, start_y + header_h), (30, 30, 30), -1)
        cv2.addWeighted(header_overlay, 0.85, frame, 0.15, 0, frame)

        col1_x = start_x + 30
        col2_x = end_x - 30
        mid_x = start_x + width // 2

        name_1 = player_names.get(1, "Player 1")
        name_2 = player_names.get(2, "Player 2")

        # player names, left/right aligned
        (n1w, _), _ = cv2.getTextSize(name_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (n2w, _), _ = cv2.getTextSize(name_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, name_1, (col1_x, start_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, name_2, (col2_x - n2w, start_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        # small centered "MATCH STATS" label
        label_text = "MATCH STATS"
        (lw, _), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(frame, label_text, (mid_x - lw // 2, start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1, cv2.LINE_AA)

        # two-tone accent line beneath header (green left / red right)
        accent_y = start_y + header_h
        cv2.line(frame, (start_x, accent_y), (mid_x, accent_y), (100, 210, 90), 2, cv2.LINE_AA)
        cv2.line(frame, (mid_x, accent_y), (end_x, accent_y), (80, 90, 220), 2, cv2.LINE_AA)

        rows = [
            ("SHOT SPEED (M/S)",     f"{player_1_shot_speed:.1f}",     f"{player_2_shot_speed:.1f}"),
            ("AVG SHOT SPEED",       f"{avg_player_1_shot_speed:.1f}", f"{avg_player_2_shot_speed:.1f}"),
            ("PLAYER SPEED",         f"{player_1_speed:.1f}",          f"{player_2_speed:.1f}"),
            ("AVG PLAYER SPEED",     f"{avg_player_1_speed:.1f}",      f"{avg_player_2_speed:.1f}"),
        ]

        row_y = start_y + header_h + 42
        row_gap = 48
        for label, val1, val2 in rows:
            # hairline divider above each row (skip the first)
            if row_y != start_y + header_h + 42:
                cv2.line(frame, (start_x + 20, row_y - row_gap + 22), (end_x - 20, row_y - row_gap + 22), (45, 45, 45), 1, cv2.LINE_AA)

            (lw2, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            cv2.putText(frame, label, (mid_x - lw2 // 2, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1, cv2.LINE_AA)

            cv2.putText(frame, val1, (col1_x, row_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
            (v2w, _), _ = cv2.getTextSize(val2, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.putText(frame, val2, (col2_x - v2w, row_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)

            row_y += row_gap

        output_video_frames[index] = frame

    return output_video_frames