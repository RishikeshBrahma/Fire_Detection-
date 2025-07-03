import cv2
import torch
import threading
import tkinter as tk
from twilio.rest import Client
import time

# Twilio configuration
account_sid = "AC6557d5c9f1921499eff6188434ac4654"
auth_token = "eba6c7af4efeafd0008b3c1674d6fef2"
twilio_phone_number = "+12316672858"
receiver_phone_number = "+918249211411"

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\iamma\\Downloads\\yolov5-fire-detection-main (1)\\yolov5-fire-detection-main\\model\\yolov5s_best.pt')

# Global variable to control webcam feed
running = False

def send_sms_alert():
    """Send SMS alert."""
    client = Client(account_sid, auth_token)
    client.messages.create(
        body="ALERT! Fire detected on your webcam feed. Take immediate action!",
        from_=twilio_phone_number,
        to=receiver_phone_number
    )
    print("SMS sent!")

def fire_detection():
    """Fire detection logic."""
    global running
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    last_sms_time = 0
    sms_interval = 60  # time in seconds between SMS alerts

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        results = model(frame)
        detections = results.pandas().xyxy[0]
        fire_detected = False

        for _, row in detections.iterrows():
            if row['name'] == 'fire':
                fire_detected = True
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw a red rectangle
                cv2.putText(frame, 'FIRE WARNING!', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        if fire_detected and (time.time() - last_sms_time > sms_interval):
            print("Fire detected! Sending SMS alert...")
            send_sms_alert()
            last_sms_time = time.time()

        cv2.imshow('Fire Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_detection():
    """Start the fire detection."""
    global running
    running = True
    threading.Thread(target=fire_detection).start()

def stop_detection():
    """Stop the fire detection."""
    global running
    running = False

# Create the Tkinter GUI
root = tk.Tk()
root.title("Fire Detection App")

start_button = tk.Button(root, text="Start Detection", command=start_detection, width=20, height=2)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Detection", command=stop_detection, width=20, height=2)
stop_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=root.quit, width=20, height=2)
exit_button.pack(pady=10)

root.mainloop()                    