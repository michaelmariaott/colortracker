from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np

colors = {
    "red": [[136,100,100],[180,255,255]]
}

def mouse_event(event):
    hsv = cv2image[event.y][eventx]
    color_range = get_color_range(hsv)
    print(color_range)
    colors["red"] = color_range

def get_color_range(h, s, v):
    custom_lower = np.array([clamp(hsv[0]-9, 0, 179),
                                clamp(hsv[1]-9, 0, 255),
                                clamp(hsv[2]-9, 0, 255)], np.uint8)
    custom_upper = np.array([clamp(hsv[0]+9, 0, 179),
                                clamp(hsv[1]+9, 0, 255),
                                clamp(hsv[2]+9, 0, 255)], np.uint8)
    return [custom_lower, custom_upper]
        


win = Tk()

win.geometry("700x350")
win.bind('<Button-1>', mouse_event)
label = Label(win)
label.grid(row=0, column=0)
cap = cv2.VideoCapture(0)

def show_frames():
    cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2HSV)

    # generate mask
    thresh_low = np.array(colors["red"][0], np.uint8)
    thresh_high = np.array(colors["red"][1], np.uint8)
    mask = cv2.inRange(cv2image, thresh_low, thresh_high)

    kernel = np.ones((5, 5), "uint8")

    mask = cv2.dilate(mask, kernel)
    result = cv2.bitwise_and(cv2image, cv2image, mask=mask) #tut uses original webcam image here (not HSV)

    # draw contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(cv2image, (x, y), 
                                    (x + w, y + h), 
                                    (255, 255, 255), 2)

    # display image
    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_HSV2RGB)
    img = Image.fromarray(cv2image)

    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    label.after(10, show_frames)

show_frames()
win.mainloop()