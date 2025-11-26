import cv2
from HandTrackingModule import HandDetector
from time import sleep
import time
import numpy as np
import cvzone
from pynput.keyboard import Controller
import ctypes
import platform

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, modelComplexity=0)
keyboard = Controller()

keys = [["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

# Add Space and Delete
buttonList.append(Button([250, 450], "SPACE", [400, 85]))
buttonList.append(Button([750, 450], "DEL", [200, 85]))

def set_transparency(window_name, alpha):
    if platform.system() != "Windows":
        return
    
    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    if hwnd:
        # GWL_EXSTYLE = -20
        # WS_EX_LAYERED = 0x00080000
        style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x00080000)
        
        # LWA_ALPHA = 0x00000002
        # Alpha should be between 0 (transparent) and 255 (opaque)
        ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, int(alpha * 255), 0x00000002)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Image", cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow("Image", 854, 480) # Set initial size to something smaller
# Set transparency to 80% (0.8)
# Note: We need to wait a tiny bit for the window to be created by OS, but usually namedWindow is synchronous enough for FindWindow
# However, sometimes it's safer to do it after a waitKey or just try immediately.
# Let's try immediately.
set_transparency("Image", 0.75)

# Pre-render the keyboard layout
# We need to capture one frame to get the size, or just hardcode 720p
success, img = cap.read()
if success:
    img = cv2.flip(img, 1)
    imgGraphics = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(imgGraphics, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgGraphics, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
else:
    print("Failed to capture initial frame")
    exit()

while True:
    success, img = cap.read()
    if not success:
        break
        
    img = cv2.flip(img, 1) # Mirror the image for easier interaction
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    
    # Blend the pre-rendered keyboard
    # This is much faster than drawing rectangles every frame
    out = img.copy()
    alpha = 0.5
    mask = imgGraphics.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgGraphics, 1 - alpha, 0)[mask]

    if lmList:
        # Draw only Index and Thumb tips for performance and clarity
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[4][1], lmList[4][2]
        cv2.circle(out, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(out, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(out, (x1, y1), (x2, y2), (255, 0, 255), 3)

        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < lmList[8][1] < x + w and y < lmList[8][2] < y + h:
                # Draw hover effect (slightly different color or just the border)
                cv2.rectangle(out, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                cv2.putText(out, button.text, (x + 20, y + 65),
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                
                # Calculate distance between index (8) and thumb (4)
                length, _, _ = detector.findDistance(8, 4, img, draw=False)

                if length < 30:
                    if button.text == "SPACE":
                        keyboard.press(' ')
                        keyboard.release(' ')
                    elif button.text == "DEL":
                        keyboard.press('\b')
                        keyboard.release('\b')
                    else:
                        keyboard.press(button.text)
                        keyboard.release(button.text)
                    
                    # Draw click effect (Green)
                    cv2.rectangle(out, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(out, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    sleep(0.15)

    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
