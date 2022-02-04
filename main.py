import cv2
from tracker import *
import math

# Create tracker object
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
height, width, _ = frame.shape
tracker = EuclideanDistTracker(0, 0, 0, height, width)
frame_count = 0
# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    avgDispX = 0
    avgDispY = 0
    ret, frame = cap.read()
    height, width, _ = frame.shape
    frame_count += 1
    # Extract Region of interest
    roi = frame[:]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.line(frame, (0, 360), (1280, 360), (0, 0, 255), 2)
    if frame_count % 5 == 0:
        if tracker.getFeatures() > 5:
            avgDispX = tracker.getDispX() / tracker.getFeatures()
            if abs(tracker.getDispX()) > 100:
                if avgDispX > 0:
                    print("LEFT")
                elif avgDispX < 0:
                    print("RIGHT")
            '''avgDispY = tracker.getDispY() / tracker.getFeatures()
            if abs(tracker.getDispY()) > 0 :
                if avgDispY > 0:
                    print("FORWARD")
                elif avgDispY < 0:
                    print("BACKWARDs")'''

        tracker.reset()


    cv2.imshow("Frame", frame)


    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()