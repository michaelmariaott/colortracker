import cv2
import numpy as np

selected_color = None
frame = None
color_index = 0

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

def click_event(event, x, y, flags, param):
    global selected_color
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_color = frame[y, x]
        detect_color(selected_color)

def trackbar_event(value):
    global color_index
    color_index = value
    print(color_index)

def detect_color(color):
    global frame
    print(color)
    if color is not None:
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
        lower_bound = np.array([clamp(hsv_color[0] - 10, 0, 179),
                                clamp(hsv_color[1] - 10, 0, 255),
                                clamp(hsv_color[2] - 10, 0, 255)])
        upper_bound = np.array([clamp(hsv_color[0] + 10, 0, 179),
                                clamp(hsv_color[1] + 10, 0, 255),
                                clamp(hsv_color[2] + 10, 0, 255)])

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        # the following 3 lines do not seem terribly necessary
        # kernel = np.ones((5, 5), "uint8")
        # mask = cv2.dilate(mask, kernel)
        # res = cv2.bitwise_and(frame, frame, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def main():
    global selected_color, frame

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Color Tracking")
    cv2.setMouseCallback("Color Tracking", click_event)
    cv2.createTrackbar("Color Index", "Color Tracking", 0, 3, trackbar_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if selected_color is not None:
            detect_color(selected_color)

        cv2.imshow("Color Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
