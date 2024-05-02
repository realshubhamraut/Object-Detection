import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(frame):
    # Convert the image to RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the image to PIL Image
    pil_image = Image.fromarray(rgb_image)
    # Perform object detection on the image
    results = model(pil_image)
    # Draw bounding boxes and labels on the image
    for *bbox, confidence, class_id in results.xyxy[0]:
        if confidence > 0.5:
            x1, y1, x2, y2 = bbox
            label = model.names[int(class_id)]
            # Get the label name from the model's class names
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def main():
    st.title("Object Detection Live Stream")
    cap = cv2.VideoCapture(1)  # Initialize the video capture object
    video_placeholder = st.empty()  # Create a placeholder for the video feed
    output_container = st.container()  # Create a container for the captured output

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        frame = detect_objects(frame)
        # Convert the image from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the live video feed
        video_placeholder.image(frame, channels="RGB", use_column_width=True)

        # Capture the output when the "Capture" button is pressed
        if st.button("Capture", key="capture_button"):
            with output_container:
                st.image(frame, channels="RGB", use_column_width=True)

    cap.release()  # Release the video capture object

if __name__ == "__main__":
    main()