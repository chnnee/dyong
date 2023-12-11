import cv2

video_path = './v2.mp4'

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(video_path)

previous_boxes = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
    new_boxes = []

    for (x, y, w, h) in boxes:
        current_box_center = (x + w // 2, y + h // 2)
        new_box = True

        for (prev_x, prev_y, prev_w, prev_h) in previous_boxes:
            previous_box_center = (prev_x + prev_w // 2, prev_y + prev_h // 2)

            if abs(current_box_center[0] - previous_box_center[0]) < 50 and abs(current_box_center[1] - previous_box_center[1]) < 50:
                new_box = False
                break
        if new_box:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            new_boxes.append((x, y, w, h))
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f'Pedestrians: {len(new_boxes)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    previous_boxes = new_boxes

    cv2.imshow('Pedestrian Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
