from utils.recognition import recognize_face

def mark_attendance(image_path):
    """Mark attendance based on recognized face."""
    person_name, confidence = recognize_face(image_path)
    print(f"Attendance marked for: {person_name} with confidence: {confidence[0][0]:.2f}")

# Example usage
mark_attendance("path_to_image.jpg")
