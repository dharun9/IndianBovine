import streamlit as st
import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("cattle.pt")


# Function to perform object detection and annotation
def perform_detection(image, confidence_threshold):
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        model.model.names[class_id]
        for class_id
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    return annotated_image

# Streamlit code
st.title("Indian Bolivian Breed Detection")

# Allow user to upload an image
uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

# Allow user to set confidence threshold
confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01)

# Button to trigger prediction
predict_button = st.button("Predict")

if predict_button and uploaded_image is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform object detection and annotation
    annotated_image = perform_detection(image, confidence_threshold)

    # Display annotated image
    st.image(annotated_image, channels="BGR", caption="Annotated Image")

# Run YOLO prediction
results = model.predict(stream=True, imgsz=512)

# Stop Streamlit app
st.stop()
