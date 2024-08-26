import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet('yolov3-tiny_training_final.weights', 'yolov3-tiny_training.cfg')

# Define the class
classes = ["Ambulance"]

# Initialize video capture
cap = cv2.VideoCapture(r'ambulance.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer to save the output video
out = cv2.VideoWriter('output_ambulance_detection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Pre-define the color and font
color = (0, 255, 0)  # Green color for bounding box
font = cv2.FONT_HERSHEY_PLAIN

# Get output layer names outside the loop
output_layers_names = net.getUnconnectedOutLayersNames()

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    height, width, _ = img.shape

    # Create a blob from the input image
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass through the YOLO network
    layerOutputs = net.forward(output_layers_names)

    # Initialize lists for detection data
    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.2:  # Confidence threshold
                # Convert YOLO coordinates to image coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression (NMS) to avoid multiple boxes for the same object
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.2, nms_threshold=0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))

            # Draw the bounding box and label on the image
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{label} {confidence}', (x, y - 10), font, 2, (255, 255, 255), 2)

    # Write the processed frame to the output video
    out.write(img)

    # Optionally display the frame
    cv2.imshow('Image', img)

    # Break loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
out.release()  # Don't forget to release the video writer
cv2.destroyAllWindows()
