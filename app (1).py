from ultralytics import YOLO
import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np
import tempfile

# Load YOLOv5 Model
MODEL_PATH = 'last.pt'  # Path to your YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)


# Helper function for inference
def detect_objects(image):
    results = model(image)
    return results


# Streamlit app setup
st.title('Face Mask Detection')
st.sidebar.title('Choose an Option')
option = st.sidebar.selectbox('Select Input Type:', ['Image', 'Video', 'Live Camera'])

if option == 'Image':
    st.header('Upload an Image')
    uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform detection
        results = detect_objects(np.array(image))
        results.render()  # Render results on the image
        detected_image = results.imgs[0]

        # Display results
        st.image(detected_image, caption='Processed Image', use_column_width=True)

elif option == 'Video':
    st.header('Upload a Video')
    uploaded_video = st.file_uploader('Choose a video...', type=['mp4', 'avi', 'mov'])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        # Display uploaded video
        st.video(video_path)

        # Process video
        st.header('Processing Video...')
        cap = cv2.VideoCapture(video_path)
        output_path = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = detect_objects(frame)
            results.render()
            out.write(results.imgs[0])

        cap.release()
        out.release()

        # Display processed video
        st.video(output_path)

elif option == 'Live Camera':
    st.header('Live Camera Detection')
    st.write('Press **`q`** to quit live detection.')

    # Open live camera
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_objects(frame)
        results.render()
        frame = results.imgs[0]

        stframe.image(frame, channels='BGR', use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
