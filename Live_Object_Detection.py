import torch
from torchvision import transforms
from PIL import Image
import cv2

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set the default camera
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the image to RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the image to PIL Image
    pil_image = Image.fromarray(rgb_image)

    # Perform object detection on the image
    results = model(pil_image)

    # Draw bounding boxes and labels on the image
    # Draw bounding boxes and labels on the image
    for *bbox, confidence, class_id in results.xyxy[0]:
        if confidence > 0.5:
            x1, y1, x2, y2 = bbox
            label = model.names[int(class_id)] # Get the label name from the model's class names
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()