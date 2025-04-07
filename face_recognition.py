from deepface import DeepFace
import cv2

def recognize_faces(video_source=0, db_path="images/"):
    """
    Perform live face recognition and identification using DeepFace.

    Parameters:
    - video_source: The source for video input (default is 0 for the webcam).
    - db_path: Path to the folder containing reference images (face database).
    """
    # Load the face database (pre-cache for optimization)
    print("Building face database...")
    DeepFace.find(img_path=None, db_path=db_path, model_name="VGG-Face")

    # Open webcam for live video feed
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Live video feed started. Press 'q' to exit.")

    while True:
        # Read frame from the live video feed
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (640, 480))

        try:
            # Perform face recognition on the current frame
            results = DeepFace.find(img_path=resized_frame, db_path=db_path, enforce_detection=False)
            if not results.empty:
                # Extract identity and display on the frame
                identified_person = results.iloc[0]['identity'].split('/')[-1]
                cv2.putText(resized_frame, f"Identified: {identified_person}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error during recognition: {e}")

        # Display the live video with identified faces
        cv2.imshow("Live Face Recognition", resized_frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Live video feed stopped.")

# Run the function with live video feed
if __name__ == "__main__":
    recognize_faces(video_source=0, db_path="images/")
