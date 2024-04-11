import cv2
from cvzone.FaceDetectionModule import FaceDetector
import cvzone
import uuid
from utils.FileOperation import write_file
import os

# Constants for offset percentages
classID = 1  # 0 is fake and 1 is real
OUTPUT_FOLDER = 'datasets/Real'  # Folder to save images
OFFSET_PERCENTAGE_WIDTH = 10  # Percentage offset for width
OFFSET_PERCENTAGE_HEIGHT = 20  # Percentage offset for height
CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence threshold for face detection
BLUR_THRESHOLD = 45  # Threshold value for blur detection
CAM_WIDTH, CAM_HEIGHT = 640, 480  # Webcam dimensions
FLOATING_POINT = 6  # Number of floating point digits for normalization
CAMBER_ID = 0  # ID of the webcam
SAVE = True  # Whether to save images
DEBUG = False  # Whether to display debug information


def main():
    # Initialize webcam and face detector
    cap = cv2.VideoCapture(CAMBER_ID)
    detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

    while True:
        # Read a frame from the webcam
        success, img = cap.read()
        if not success:
            break

        # Create a copy of the image for output
        imgOut = img.copy()

        # Get image dimensions
        img_h, img_w, _ = img.shape

        # Detect faces in the image
        img, bboxs = detector.findFaces(img, draw=False)
        blur_list = []  # List to store blur detection results
        info_list = []  # List to store face information

        # Process each detected face
        for bbox in bboxs[:1]:
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = bbox['score'][0]

            # Check if face detection confidence meets threshold
            if score < CONFIDENCE_THRESHOLD:
                continue

            # Add offset to the detected face region
            offsetW = (OFFSET_PERCENTAGE_WIDTH / 100) * w
            offsetH = (OFFSET_PERCENTAGE_HEIGHT / 100) * h
            x, y, w, h = max(int(x - offsetW), 0), max(int(y - offsetH * 3), 0), max(int(w + offsetW * 2), 0), max(
                int(h + offsetH * 3.5), 0)

            # Extract the face region
            imgFace = img[y:y + h, x:x + w]

            # Calculate blur value
            blurValue = cv2.Laplacian(imgFace, cv2.CV_64F).var()

            # Append result of blur detection to blur_list
            blur_list.append(blurValue > BLUR_THRESHOLD)

            # Normalize face coordinates and dimensions
            cx, cy = x + w / 2, y + h / 2
            xcn, ycn = min(round(cx / img_w, FLOATING_POINT), 1), min(round(cy / img_h, FLOATING_POINT), 1)
            wn, hn = min(round(w / img_w, FLOATING_POINT), 1), min(round(h / img_h, FLOATING_POINT), 1)

            # Append face information to info_list
            info_list.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")  # Yolo format
            # Draw face data on the output image
            cv2.circle(imgOut, center, 2, (255, 0, 255), cv2.FILLED)
            cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur:{int(blurValue)}', (x, y - 10), scale=1,
                               thickness=2)
            cvzone.cornerRect(imgOut, (x, y, w, h))

            # Display blurred face
            cv2.imshow("Blurred Face", imgFace)

        # Display the annotated image
        cv2.imshow("Data Collection", imgOut)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save images if enabled and all detected faces are blurred
        if SAVE and all(blur_list) and len(blur_list) > 0:
            # get len of the location
            total_files= len(os.listdir(OUTPUT_FOLDER))
            if total_files > 896:
                break
            filename = str(uuid.uuid4())
            cv2.imwrite(f'{OUTPUT_FOLDER}/{filename}.jpg', img)
            # save Label
            for info in info_list:
                write_file(f'{OUTPUT_FOLDER}/{filename}.txt', info)

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
