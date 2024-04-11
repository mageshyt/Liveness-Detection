import cv2
from cvzone.FaceDetectionModule import FaceDetector
import cvzone

#######################################
# Constants for offset percentages
OFFSET_PERCENTAGE_WIDTH = 10
OFFSET_PERCENTAGE_HEIGHT = 20
CONFIDENCE_THRESHOLD = 0.8
CAM_WIDTH ,CAM_HEIGHT= 640,480
FLOATING_POINT = 6
CAMBERA_ID = 0

#######################################
def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(CAMBERA_ID)

    # Initialize the face detector
    detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

    while True:
        # Read the current frame from the webcam
        success, img = cap.read()

        # Detect faces in the image
        img, bboxs = detector.findFaces(img, draw=False)

        # Process each detected face
        for bbox in bboxs[:1]:  # Process only the first detected face
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)
            # --------------------- check score---------------------
            if score < CONFIDENCE_THRESHOLD:
                continue

            # -------------------Add offset to the detected face region-------------------
            offsetW = (OFFSET_PERCENTAGE_WIDTH / 100) * w
            offsetH = (OFFSET_PERCENTAGE_HEIGHT / 100) * h
            x = max(int(x - offsetW), 0)
            y = max(int(y - offsetH * 3), 0)
            w = max(int(w + offsetW * 2), 0)
            h = max(int(h + offsetH * 3.5), 0)

            # Extract the face region
            imgFace = img[y:y+h, x:x+w]

            # Calculate blur value
            blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

            # --------------------Normalizer Value--------------------
            img_h, img_w ,_= imgFace.shape # height, width , channel
            # center points
            cx,cy=x+w//2,y+h//2
            # normalize x and y
            xcn,ycn=round(cx/img_w,FLOATING_POINT),round(cy/img_h,FLOATING_POINT)

            # --------------------Draw face data--------------------
            cv2.circle(img, center, 2, (255, 0, 255), cv2.FILLED)
            cvzone.putTextRect(img, f'Blur:{blurValue}', (x, y - 10))
            cvzone.cornerRect(img, (x, y, w, h))

            # Display blurred face
            cv2.imshow("Blurred Face", imgFace)

        # Display the image
        cv2.imshow("Data Collection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
