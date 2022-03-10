import cv2
import numpy as np

video = cv2.VideoCapture("drivingVideo.mp4")

while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture("drivingVideo.mp4")
        continue

    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    low_white = np.array([0, 0, 240])
    up_white = np.array([255, 15, 255])
    mask = cv2.inRange(hsv, low_white, up_white)
    
    edges = cv2.Canny(mask, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=50)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 10)

    cv2.imshow("Frame", frame)
    cv2.imshow("Edges", edges)

    key = cv2.waitKey(5)

video.release()
cv2.destroyAllWindows()
