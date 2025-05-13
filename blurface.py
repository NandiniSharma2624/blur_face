import cv2
import mediapipe as mp

# Load the mediapipe face detection model
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    results = face_detection.process(frame_rgb)

    # If faces are detected, apply blur to each face
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Extract the face ROI (Region of Interest)
            face_roi = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
             # Apply Gaussian blur to the face ROI
            blur_face = cv2.GaussianBlur(face_roi, (99, 99), 30)

            # Replace the face ROI with the blurred face
            frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = blur_face

    # Display the output
    cv2.imshow('Blurred Faces', frame)

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()