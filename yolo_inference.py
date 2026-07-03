from ultralytics import YOLO

model = YOLO("yolo11l")  # Load a pretrained YOLOv11 model


result = model.track('/Users/peter/Desktop/tennis_analysis/input_videos/input_video.mp4', conf=0.2, save = True) #variable for output

# print(result)
# print("boxes:")
# for box in result[0].boxes:
#     print(box)




