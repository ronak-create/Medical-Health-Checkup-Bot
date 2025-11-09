import cv2
from ultralytics import YOLO
import time
import numpy as np

def main():
    # Initialize YOLO model
    model = YOLO('yolov12s.pt')  # Using YOLOv8 nano model for better speed
    
    # Initialize video capture (0 for webcam)
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize variables for FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Read frame from camera
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        # Run YOLO detection
        results = model(frame, stream=True)
        
        # Process detection results
        for result in results:
            boxes = result.boxes
            
            # Visualize detections
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class name and confidence
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = model.names[cls]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f'{class_name} {conf:.2f}'
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 30:  # Update FPS every 30 frames
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
            
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('YOLOv8 Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()