
import cv2
import os

def save_frames(video_path, output_folder, image_format='jpg'):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get some properties from the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Loop through each frame
    for i in range(frame_count):
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        # Construct the output file path
        filename = f"frame_{i:04d}.{image_format}"
        output_path = os.path.join(output_folder, filename)

        # Save the frame as an image
        cv2.imwrite(output_path, frame)

        # Print progress
        print(f"Saved frame {i+1}/{frame_count}")

    # Release the video capture object
    cap.release()

    print("All frames saved successfully.")

# Example usage
video_file = "./Data/ball_videos/IMG_9078.MOV"
output_folder = "output_frames"
image_format = 'jpg'  # Change to 'png' if you prefer PNG format

save_frames(video_file, output_folder, image_format)
