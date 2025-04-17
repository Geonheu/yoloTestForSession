import datetime
import cv2
from ultralytics import YOLO
import yaml
from deep_sort_realtime.deepsort_tracker import DeepSort

# DeepSort 사용, max_age보다 오래 탐지되지 않으면 추적 X
tracker = DeepSort(max_age=50)

CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# coco128 Dataset 클래스 목록 가져오기
yaml_path = '/Users/jogeonhui/Documents/LikeLion/First_Session/coco128.txt'
with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)

class_list = data.get('names', [])
# print(f"클래스 목록 : {class_list}")

# coco128 = open('./yolov8_pretrained/coco128.txt', 'r')
# data = coco128.read()
# class_list = data.split('\n')
# coco128.close()

# 모델 불러오기 (자동 다운로드)
model = YOLO('./yolov8_pretrained/yolov8n.pt')

cap = cv2.VideoCapture(0)  # 내장 웹캠 접근
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 웹캠 실행
if not cap.isOpened():
    print('fail')

while True:
    start = datetime.datetime.now()

    ret, frame = cap.read()
    # 데이터가 제대로 수신되는 지 확인
    if not ret:
        print("Cam Error")
        break
    print(f"Type: {type(frame)}, Shape: {frame.shape}")

    detection = model.predict(source=[frame], save=False)[0]
    results = []

    for data in detection.boxes.data.tolist():
        confidence = float(data[4])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        label = int(data[5])
        # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        # cv2.putText(frame, class_list[label]+' '+str(round(confidence, 2)) +'%', (xmin, ymin), cv2.FONT_ITALIC, 1, WHITE, 2)

        results.append([[xmin, ymin, xmax-xmin, ymax-ymin], confidence, label])

    tracks = tracker.update_tracks(results, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
    
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.rectangle(frame, (xmin, ymin-20), (xmin+20, ymin), GREEN, -1)
        cv2.putText(frame, str(track_id), (xmin+5, ymin-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

    end = datetime.datetime.now()

    total = (end - start).total_seconds()
    # print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

    fps = f'FPS: {1 / total:.2f}'
    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

