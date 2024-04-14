import time
import cv2
import cvzone
from ultralytics import YOLO

confidence = 0.6


def main():
    cap = cv2.VideoCapture(0)  # For Webcam
    cap.set(3, 640)
    cap.set(4, 480)

    model = YOLO("runs/detect/train/weights/best.pt")
    test_img=cv2.imread("datasets/Real/0a2bbbab-a3ea-46cd-a6c8-0e7cbfb554bf.jpg")

    # predict
    # results = model.predict(test_img,save=True)

    # print(results)

    classNames = ["fake", "real"]

    prev_frame_time = 0
    new_frame_time = 0

    while True:

        success, img = cap.read()
        if not success:
            print("Error reading frame from camera")
            break

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        results = model.predict(img, save=False,verbose=False)
        for r in results:
            print(len(r.boxes))
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = round(float(box.conf[0]), 2)  # Convert Tensor to float and then round
                cls = int(box.cls[0])
                print("Confidence: ", conf, "Class: ", classNames[cls],"Conf",box.conf,"Cls",box.cls)
                if conf > confidence:
                    color = (0, 255, 0) if classNames[cls] == 'real' else (0, 0, 255)
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                    cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                       (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color, colorB=color)

        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Large model", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
