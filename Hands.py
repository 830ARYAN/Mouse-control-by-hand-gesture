import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

screen_width, screen_height = pyautogui.size()


camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

smoothing_factor = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0


position_buffer = deque(maxlen=smoothing_factor)


click_cooldown = 0
cooldown_period = 10


prev_time = 0
fps = 0

print("Hand Tracking Mouse Control Started")
print("Press 'ESC' to exit")

while True:
  
    success, image = camera.read()
    if not success:
        print("Failed to read from camera")
        break
    
    
    current_time = cv2.getTickCount()
    time_diff = (current_time - prev_time) / cv2.getTickFrequency()
    if time_diff > 0:
        fps = 1.0 / time_diff
    prev_time = current_time
    
   
    image = cv2.flip(image, 1)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    results = hands.process(rgb_image)
    
    
    if click_cooldown > 0:
        click_cooldown -= 1
    
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
      
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
    
        index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        h, w, c = image.shape
        x_index = int(index_finger.x * w)
        y_index = int(index_finger.y * h)
        
        cv2.circle(image, (x_index, y_index), 10, (0, 255, 255), -1)
        cv2.circle(image, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 10, (0, 255, 0), -1)
        cv2.circle(image, (int(middle_tip.x * w), int(middle_tip.y * h)), 10, (255, 0, 0), -1)
        
       
        mouse_x = np.interp(index_finger.x, [0, 1], [0, screen_width])
        mouse_y = np.interp(index_finger.y, [0, 1], [0, screen_height])
        
     
        position_buffer.append((mouse_x, mouse_y))
        
        
        if len(position_buffer) > 0:
            smooth_x = sum(pos[0] for pos in position_buffer) / len(position_buffer)
            smooth_y = sum(pos[1] for pos in position_buffer) / len(position_buffer)
            
            
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = smooth_x, smooth_y
            
            curr_x = prev_x + (smooth_x - prev_x) / smoothing_factor
            curr_y = prev_y + (smooth_y - prev_y) / smoothing_factor
            
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y
        
        
        dist_thumb_index = ((thumb_tip.x - index_finger.x)**2 + 
                           (thumb_tip.y - index_finger.y)**2)**0.5
        
        
        dist_index_middle = ((middle_tip.x - index_finger.x)**2 + 
                            (middle_tip.y - index_finger.y)**2)**0.5
        
        
        if dist_thumb_index < 0.05 and click_cooldown == 0:
            pyautogui.click()
            click_cooldown = cooldown_period
            cv2.putText(image, "Left Click", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
        
      
        if dist_index_middle < 0.05 and click_cooldown == 0:
            pyautogui.rightClick()
            click_cooldown = cooldown_period
            cv2.putText(image, "Right Click", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
   
    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 255, 0), 2)
    cv2.putText(image, "Thumb+Index: Left Click", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 255, 255), 2)
    cv2.putText(image, "Index+Middle: Right Click", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 255, 255), 2)
    cv2.putText(image, "Press 'ESC' to exit", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 255, 255), 2)
    
 
    cv2.imshow("Hand Tracking Mouse Control", image)
    
 
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  
        break


camera.release()
cv2.destroyAllWindows()
print("Application closed successfully")
