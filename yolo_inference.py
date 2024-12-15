from ultralytics import YOLO

model = YOLO('Train/models/best.pt')

results = model.predict('path/to/video.mp4', save=True)
print(results[0])
print('==========================')
for box in results[0].boxes:
    print(box)