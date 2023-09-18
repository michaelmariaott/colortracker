import cv2
import numpy as np
from pythonosc import udp_client

max_colors = 4
selected_colors = [None] * max_colors
color_areas = [0] * max_colors
frame = None
color_index = 0


def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

def click_event(event, x, y, flags, param):
    global selected_colors, color_index
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_color = frame[y, x]
        selected_colors[color_index] = selected_color
        detect_colors(selected_colors)

def trackbar_event(value):
    global color_index
    color_index = value
    print(color_index)

def detect_colors(colors, color_areas):
    global frame
    for index, color in enumerate(colors):
        if color is not None:
            hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
            lower_bound = np.array([clamp(hsv_color[0] - 10, 0, 179),
                                    clamp(hsv_color[1] - 30, 0, 255),
                                    clamp(hsv_color[2] - 30, 0, 255)])
            upper_bound = np.array([clamp(hsv_color[0] + 10, 0, 179),
                                    clamp(hsv_color[1] + 30, 0, 255),
                                    clamp(hsv_color[2] + 30, 0, 255)])

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
            # the following 3 lines do not seem necessary
            # kernel = np.ones((5, 5), "uint8")
            # mask = cv2.dilate(mask, kernel)
            # res = cv2.bitwise_and(frame, frame, mask=mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            area_sum = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (int(color[0]), int(color[1]), int(color[2])), 2)
                    area_sum += area
            color_areas[index] = area_sum
            print(color_areas)

def send_osc(osc_client, color_areas):
    areas_sum = sum(color_areas)
    if areas_sum > 0:
        for index, color_area in enumerate(color_areas):
            osc_client.send_message("/colors/absolute/"+str(index), color_area)
            osc_client.send_message("/colors/relative/"+str(index), color_area/areas_sum)
                    

def main():
    global selected_colors, frame, color_areas

    osc_client = udp_client.SimpleUDPClient("127.0.0.1", 5555)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Color Tracking")
    cv2.setMouseCallback("Color Tracking", click_event)
    cv2.createTrackbar("Color Nr.", "Color Tracking", 0, max_colors - 1, trackbar_event)


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if selected_colors:
            detect_colors(selected_colors, color_areas)
            send_osc(osc_client, color_areas)

        cv2.imshow("Color Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
