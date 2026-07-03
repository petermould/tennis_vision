import cv2
import sys
sys.path.append('../')
import constants
from utils.conversions import convert_metres_to_pixel_distance, convert_pixel_distance_to_metres
import numpy as np
from utils.bbox_utils import get_center_of_bbox, measure_distance, get_foot_position, get_closest_keypoint_index, get_height_of_bbox, measure_xy_distance, get_center_of_bbox

class MiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.buffer = 50
        self.padding_court = 20

        # Court width available inside the box, after padding
        self.court_drawing_width = self.drawing_rectangle_width - 2 * self.padding_court

        # Derive height from the real court's length:width ratio
        court_height_pixels = int(
            self.court_drawing_width * (constants.HALF_COURT_LINE_HEIGHT * 2 / constants.DOUBLE_LINE_WIDTH)
        )
        self.drawing_rectangle_height = court_height_pixels + 2 * self.padding_court

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def convert_metres_to_pixels(self, metres):
        return convert_metres_to_pixel_distance(metres,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width)

    def set_court_drawing_key_points(self):
        drawing_key_points = [0] * 28

        # point 0
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_metres_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        # point 3
        drawing_key_points[6] = self.court_start_x + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]

        # point 4
        drawing_key_points[8] = drawing_key_points[0] + self.convert_metres_to_pixels(constants.DOUBLE_ALLEY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1]
        # point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_metres_to_pixels(constants.DOUBLE_ALLEY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5]
        # point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_metres_to_pixels(constants.DOUBLE_ALLEY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3]
        # point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_metres_to_pixels(constants.DOUBLE_ALLEY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7]
        # point 8
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_metres_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_metres_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17]
        # point 10
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_metres_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 11
        drawing_key_points[22] = drawing_key_points[20] + self.convert_metres_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21]
        # point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18]) / 2)
        drawing_key_points[25] = drawing_key_points[17]
        # point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22]) / 2)
        drawing_key_points[27] = drawing_key_points[21]

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            (0, 1),
            (8, 9),
            (10, 11),
            (10, 11),
            (2, 3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
    # court_drawing_width already set in __init__, no need to recalculate here

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self, frame):
    # Fill court background (light green, matches mini-court box)
        court_corners = np.array([
            [self.drawing_key_points[0], self.drawing_key_points[1]],
            [self.drawing_key_points[2], self.drawing_key_points[3]],
            [self.drawing_key_points[6], self.drawing_key_points[7]],
            [self.drawing_key_points[4], self.drawing_key_points[5]],
        ], np.int32)
        cv2.fillPoly(frame, [court_corners], (60, 140, 60))  # muted green

    # Draw lines (white, anti-aliased)
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw net (distinct color, slightly thicker)
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        cv2.line(frame, net_start_point, net_end_point, (200, 200, 200), 3, cv2.LINE_AA)

    # Keypoints — small, subtle, anti-aliased (toggle off once you trust calibration)
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)

        return frame

    def draw_background_rectange(self, frame):
        shapes = np.zeros_like(frame, np.uint8)

        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        

        return out

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectange(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
    
    def get_mini_court_coordinates(self, 
                                   object_position, 
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters):
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        #convert pixel distance to m
        distance_from_keypoint_x_metres = convert_pixel_distance_to_metres(distance_from_keypoint_x_pixels, 
                                                                           player_height_in_meters, 
                                                                           player_height_in_pixels)
        distance_from_keypoint_y_metres = convert_pixel_distance_to_metres(distance_from_keypoint_y_pixels, 
                                                                           player_height_in_meters, 
                                                                           player_height_in_pixels)
        
        #convert to mini
        mini_court_x_distance_pixels = self.convert_metres_to_pixels(distance_from_keypoint_x_metres)
        mini_court_y_distance_pixels = self.convert_metres_to_pixels(distance_from_keypoint_y_metres)
        closest_mini_court_key_point = (self.drawing_key_points[closest_key_point_index*2], 
                                        self.drawing_key_points[closest_key_point_index*2+1])
        mini_court_player_position = (closest_mini_court_key_point[0] + mini_court_x_distance_pixels, 
                                      closest_mini_court_key_point[1] + mini_court_y_distance_pixels)
        
        return mini_court_player_position
        

        
    
    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        player_ids = sorted(player_boxes[0].keys())
        player_heights = {
            player_ids[0]: constants.PLAYER_1_HEIGHT_METRES,
            player_ids[1]: constants.PLAYER_2_HEIGHT_METRES
        }

        output_player_boxes = []
        output_ball_boxes = []
        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_bbox(ball_box)
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_position, get_center_of_bbox(player_bbox[x])))

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                closest_key_point_index = get_closest_keypoint_index(foot_position, original_court_key_points, [0, 2, 12, 13])
                closest_key_point = (original_court_key_points[closest_key_point_index*2],
                                    original_court_key_points[closest_key_point_index*2+1])

                frame_index_min = max(0, frame_num-20)
                frame_index_max = min(len(player_boxes), frame_num+50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in range(frame_index_min, frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)

                mini_court_player_position = self.get_mini_court_coordinates(
                    foot_position,
                    closest_key_point,
                    closest_key_point_index,
                    max_player_height_in_pixels,
                    player_heights[player_id]
                )
                output_player_bboxes_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    closest_key_point_index = get_closest_keypoint_index(ball_position, original_court_key_points, [0, 2, 12, 13])
                    closest_key_point = (original_court_key_points[closest_key_point_index*2],
                                        original_court_key_points[closest_key_point_index*2+1])
                    mini_court_ball_position = self.get_mini_court_coordinates(
                        ball_position,
                        closest_key_point,
                        closest_key_point_index,
                        max_player_height_in_pixels,
                        player_heights[player_id]
                    )
                    output_ball_boxes.append({1: mini_court_ball_position})  # ← moved inside the if

            output_player_boxes.append(output_player_bboxes_dict)

        return output_player_boxes, output_ball_boxes
        
    def draw_points_on_mini_court(self,frames,postions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in postions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames

    def draw_ball_trail(self, frames, ball_mini_court_detections, trail_length=10):
        for frame_num, frame in enumerate(frames):
            start_idx = max(0, frame_num - trail_length)
            trail_positions = ball_mini_court_detections[start_idx:frame_num + 1]

            for i, position_dict in enumerate(trail_positions):
                if 1 not in position_dict:
                    continue
                x, y = position_dict[1]

                # older points = smaller and more faded
                age = len(trail_positions) - i  # 1 = newest, trail_length = oldest
                alpha = max(0.15, 1.0 - (age / trail_length))
                radius = max(2, int(6 * alpha))

                overlay = frame.copy()
                cv2.circle(overlay, (int(x), int(y)), radius, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frames

