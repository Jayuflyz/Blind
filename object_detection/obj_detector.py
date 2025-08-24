import cv2

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
filename = 'labels.txt'
with open(filename, 'r') as spt:
    classLabels = spt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# ✅ Open live webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# ✅ Read first frame for dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read from camera.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output_live_video.mp4', fourcc, 25, (frame.shape[1], frame.shape[0]))

font = cv2.FONT_HERSHEY_PLAIN

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        classIndex, confidence, bbox = model.detect(frame, confThreshold=0.70)
        if len(classIndex) != 0:
            for classInd, boxes in zip(classIndex.flatten(), bbox):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                if classInd - 1 < len(classLabels):
                    cv2.putText(frame, classLabels[classInd - 1], (boxes[0] + 10, boxes[1] + 40),
                                font, fontScale=1, color=(0, 255, 0), thickness=2)

        video.write(frame)
        cv2.imshow('Live Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print("Exception occurred:", e)

cap.release()
video.release()
cv2.destroyAllWindows()
