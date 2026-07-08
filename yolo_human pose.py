
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

def main():
    cap = cv2.VideoCapture(0)
    model = YOLO("yolov8n-pose.pt")
    cv2.namedWindow("YOLOv8 Pose")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        body_parts = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]
        
        if results.keypoints is not None:
            kpts = results.keypoints.xy.cpu().numpy()
            for person_kpts in kpts:
                for idx, (x, y) in enumerate(person_kpts):
                    if x == 0 and y == 0:
                        continue
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    if idx < len(body_parts):
                        part_name = body_parts[idx]
                        cv2.putText(frame, part_name, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


        cv2.imshow("YOLOv8 Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
