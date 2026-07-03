from utils.video_utils import read_video, save_video
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_line_detector.court_line_detector import CourtLineDetector
import cv2
from mini_court.mini_court import MiniCourt
from utils.bbox_utils import measure_distance
from copy import deepcopy
import pandas as pd
from utils.player_stats_drawer_utils import draw_player_stats
from utils.live_stats_utils import draw_live_stat_tags
from utils.conversions import convert_pixel_distance_to_metres, convert_metres_to_pixel_distance
import constants


def main():
    input_video_path = "input_videos/video.mp4"
    video_frames = read_video(input_video_path)

    # get real fps from the source video (was previously hardcoded — caused deflated speeds)
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    player_tracker = PlayerTracker(model_path="yolo11l.pt")
    ball_tracker = BallTracker(model_path="/Users/peter/Desktop/tennis_analysis/models/best-3.pt")

    player_detections = player_tracker.detect_frames(
        video_frames, read_from_stub=True,
        stub_path="/Users/peter/Desktop/tennis_analysis/tracker_stubs/player_detections.pkl"
    )
    ball_detections = ball_tracker.detect_frames(
        video_frames, read_from_stub=True,
        stub_path="/Users/peter/Desktop/tennis_analysis/tracker_stubs/ball_detections.pkl"
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # court line detector
    court_model_path = "/Users/peter/Desktop/tennis_analysis/models/keypoints_model_trained.pth"
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players (filters out officials/ball kids, keeps the 2 real players)
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # minicourt
    mini_court = MiniCourt(video_frames[0])

    # ball tracker for shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # convert positions to mini court coordinates
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints
    )

    # map real track IDs (e.g. 1, 4) to fixed stats labels 1 and 2
    player_ids = sorted(player_mini_court_detections[0].keys())
    id_to_label = {player_ids[0]: 1, player_ids[1]: 2}

    player_stats_data = [{
        'frame_numb': 0,
        'player_1_number_of_shots': 0,
        "player_1_total_shot_speeds": 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        "player_2_total_shot_speeds": 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]

    # collected for the on-court floating shot tags
    shot_events = []

    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / fps

        # distance covered by ball
        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1]
        )
        distance_covered_by_ball_metres = convert_pixel_distance_to_metres(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )
        speed_of_ball_shot = distance_covered_by_ball_metres / ball_shot_time_in_seconds

        # player who shot the ball (real track ID)
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(
            player_positions.keys(),
            key=lambda player_id: measure_distance(
                player_positions[player_id], ball_mini_court_detections[start_frame][1]
            )
        )

        # opponent (real track ID) — the other one of the two real player IDs
        opponent_player_id = player_ids[0] if player_shot_ball == player_ids[1] else player_ids[1]

        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id]
        )
        distance_covered_by_opponent_metres = convert_pixel_distance_to_metres(
            distance_covered_by_opponent_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )
        speed_of_opponent = distance_covered_by_opponent_metres / ball_shot_time_in_seconds

        # convert real track IDs to fixed labels 1/2 for the stats dict keys
        shot_label = id_to_label[player_shot_ball]
        opponent_label = id_to_label[opponent_player_id]

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_numb'] = start_frame
        current_player_stats[f'player_{shot_label}_number_of_shots'] += 1
        current_player_stats[f'player_{shot_label}_total_shot_speeds'] += speed_of_ball_shot
        current_player_stats[f'player_{shot_label}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_label}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_label}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

        # record this shot event (real track id) for the floating tag drawer
        shot_events.append({
            'frame': start_frame,
            'player_id': player_shot_ball,
            'speed': speed_of_ball_shot
        })

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_numb': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_numb', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = (
        player_stats_data_df['player_1_total_shot_speeds'] / player_stats_data_df['player_1_number_of_shots']
    )
    player_stats_data_df['player_2_average_shot_speed'] = (
        player_stats_data_df['player_2_total_shot_speeds'] / player_stats_data_df['player_2_number_of_shots']
    )
    player_stats_data_df['player_1_average_player_speed'] = (
        player_stats_data_df['player_1_total_player_speed'] / (player_stats_data_df['player_2_number_of_shots'] + 1)
    )
    player_stats_data_df['player_2_average_player_speed'] = (
        player_stats_data_df['player_2_total_player_speed'] / (player_stats_data_df['player_1_number_of_shots'] + 1)
    )

    # draw output — order matters: draw base layers first, panels/overlays after
    output_video_frames = player_tracker.draw_bbox(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # mini-court panel must be drawn before anything gets drawn onto it
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_ball_trail(output_video_frames, ball_mini_court_detections, trail_length=10)

    # summary stats panel (left-middle, running totals/averages)
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    # on-court floating tags: player speed pill + fading shot speed tag after each hit
    output_video_frames = draw_live_stat_tags(
        output_video_frames, player_detections, shot_events, player_stats_data_df,
        id_to_label, fps=fps, fade_seconds=1.5
    )

    # draw frame number in top left corner
    for i in range(len(output_video_frames)):
        cv2.putText(output_video_frames[i], f"Frame: {i}", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()