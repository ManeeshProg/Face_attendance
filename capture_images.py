import cv2
import os

def capture_images(person_name, dataset_path="data/dataset"):
    """Capture images from webcam and save them to the specified folder."""
    # Create a folder for the person if it doesn't exist
    person_folder = os.path.join(dataset_path, person_name)
    os.makedirs(person_folder, exist_ok=True)

    # Initialize webcam (use 0 for default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"Starting to capture images for {person_name}...")

    img_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the captured frame
        cv2.imshow("Capture Image - Press 'q' to quit", frame)

        # Save the frame when 's' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            img_count += 1
            img_path = os.path.join(person_folder, f"{person_name}_{img_count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Image {img_count} saved as {img_path}")

        # Exit loop if 'q' is pressed
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting image capture.")
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    person_name = input("Enter the name of the person: ")
    capture_images(person_name)
