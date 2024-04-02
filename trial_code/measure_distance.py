# import cv2
# import numpy as np

# def measure_ball_distance(ball_center, focal_length, camera_matrix, pixel_size_in_cm):
#     """
#     Measures the distance from the camera to the ball in centimeters.

#     Args:
#         ball_center (tuple): Center of the ball as (x, y) coordinates in pixels.
#         focal_length (float): Focal length in pixels.
#         camera_matrix (numpy.ndarray): Camera matrix (3x3).
#         pixel_size_in_cm (float): Size of one pixel in centimeters.

#     Returns:
#         float: Distance from the ball to the camera in centimeters.
#     """

#     # Calculate the depth using the camera matrix, ball center, and focal length.
#     z = (camera_matrix[0, 0] * focal_length * pixel_size_in_cm) / ball_center[0]

#     return z

# # Initialize the camera
# camera = cv2.VideoCapture(0)

# # Define the lower and upper bounds of the red and blue colors
# redLower = np.array([0, 0, 200], dtype=np.uint8)
# redUpper = np.array([10, 10, 255], dtype=np.uint8)
# blueLower = np.array([100, 100, 0], dtype=np.uint8)
# blueUpper = np.array([150, 150, 255], dtype=np.uint8)

# # Define the camera matrix and distortion coefficients.
# # Assuming a simple camera setup, using the identity matrix.
# camera_matrix = np.identity(3, dtype=np.float32)

# # Set your desired pixel size (width or height) in centimeters
# pixel_size_in_cm = 0.0028  # This represents a 720p image (1280x720) with width of around 19.2 cm.

# while True:
#     # Capture the current frame
#     ret, frame = camera.read()

#     # Convert the frame to HSV color space
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Construct a mask for the red and blue colors
#     redMask = cv2.inRange(hsv, redLower, redUpper)
#     blueMask = cv2.inRange(hsv, blueLower, blueUpper)

#     # Perform morphological operations to reduce noise
#     redMask = cv2.morphologyEx(redMask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
#     blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

#     # Find the contours of the mask
#     redContours = cv2.findContours(redMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     blueContours = cv2.findContours(blueMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Compute the centroids of the red and blue balls
#     redBallCenter = None
#     blueBallCenter = None

#     if len(redContours[0]) > 0:
#         redCnt = max(redContours[0], key=cv2.contourArea)
#         redMoment = cv2.moments(redCnt)
#         redBallCenter = (int(redMoment["m10"] / redMoment["m00"]),
#                 int(redMoment["m01"] / redMoment["m00"]))

#     if len(blueContours[0]) > 0:
#         blueCnt = max(blueContours[0], key=cv2.contourArea)
#         blueMoment = cv2.moments(blueCnt)
#         blueBallCenter = (int(blueMoment["m10"] / blueMoment["m00"]),
#                            int(blueMoment["m01"] / blueMoment["m00"]))

#     # Draw the centroids of the red and blue balls
#     if redBallCenter is not None:
#         cv2.circle(frame, redBallCenter, 5, (0, 0, 255), -1)

#         # Measure the distance
#         red_distance = int(redMoment["m01"] / redMoment["m00"])

#         # Measure the distance
#         red_ball_dist = measure_ball_distance(
#             ball_center=redBallCenter,
#             focal_length=320,  # Assuming the center of the ball is along the image's width
#             camera_matrix=camera_matrix,
#             pixel_size_in_cm=pixel_size_in_cm
#         )

#         print(f"Red Ball Distance: {red_ball_dist} cm")

#     if len(blueContours[0]) > 0:
#         blueCnt = max(blueContours[0], key=cv2.contourArea)
#         blueMoment = cv2.moments(blueCnt)
#         blueBallCenter = (int(blueMoment["m10"] / blueMoment["m00"]),
#                            int(blueMoment["m01"] / blueMoment["m00"]))

#         # Measure the distance
#         blue_ball_dist = measure_ball_distance(
#             ball_center=blueBallCenter,
#             focal_length=240,  # Assuming the center of the ball is along the image's height
#             camera_matrix=camera_matrix,
#             pixel_size_in_cm=pixel_size_in_cm
#         )

#         print(f"Blue Ball Distance: {blue_ball_dist} cm")

#     # Draw the centroids of the red and blue balls
#     if redBallCenter is not None:
#         cv2.circle(frame, redBallCenter, 5, (0, 0, 255), -1)

#     if blueBallCenter is not None:
#         cv2.circle(frame, blueBallCenter, 5, (255, 0, 0), -1)

#     cv2.imshow("Frame", frame)

#     # Exit the loop by pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# camera.release()
# cv2.destroyAllWindows()














import cv2
import numpy as np
import os

def measure_ball_distance(ball_center, focal_length, camera_matrix, pixel_size_in_cm):
    """
    Measures the distance from the camera to the ball in centimeters.

    Args:
        ball_center (tuple): Center of the ball as (x, y) coordinates in pixels.
        focal_length (float): Focal length in pixels.
        camera_matrix (numpy.ndarray): Camera matrix (3x3).
        pixel_size_in_cm (float): Size of one pixel in centimeters.

    Returns:
        float: Distance from the ball to the camera in centimeters.
    """

    # Calculate the depth using the camera matrix, ball center, and focal length.
    z = (camera_matrix[0, 0] * focal_length * pixel_size_in_cm) / ball_center[0]

    return z

# Load the image
image_path = "./Data/rb24.jpg"
frame = cv2.imread(image_path)

# Convert the frame to HSV color space
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds of the red and blue colors
redLower = np.array([0, 0, 200], dtype=np.uint8)
redUpper = np.array([10, 10, 255], dtype=np.uint8)
blueLower = np.array([100, 100, 0], dtype=np.uint8)
blueUpper = np.array([150, 150, 255], dtype=np.uint8)

# Define the camera matrix and distortion coefficients.
# Assuming a simple camera setup, using the identity matrix.
camera_matrix = np.identity(3, dtype=np.float32)

# Set your desired pixel size (width or height) in centimeters
pixel_size_in_cm = 0.0028  # This represents a 720p image (1280x720) with width of around 19.2 cm.

# Construct a mask for the red and blue colors
redMask = cv2.inRange(hsv, redLower, redUpper)
blueMask = cv2.inRange(hsv, blueLower, blueUpper)

# Perform morphological operations to reduce noise
redMask = cv2.morphologyEx(redMask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

# Find the contours of the mask
redContours = cv2.findContours(redMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
blueContours = cv2.findContours(blueMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Compute the centroids of the red and blue balls
redBallCenter = None
blueBallCenter = None

if len(redContours[0]) > 0:
    redCnt = max(redContours[0], key=cv2.contourArea)
    redMoment = cv2.moments(redCnt)
    redBallCenter = (int(redMoment["m10"] / redMoment["m00"]),
                     int(redMoment["m01"] / redMoment["m00"]))

if len(blueContours[0]) > 0:
    blueCnt = max(blueContours[0], key=cv2.contourArea)
    blueMoment = cv2.moments(blueCnt)
    blueBallCenter = (int(blueMoment["m10"] / blueMoment["m00"]),
                      int(blueMoment["m01"] / blueMoment["m00"]))

# Draw the centroids of the red and blue balls and mark distance
font = cv2.FONT_HERSHEY_SIMPLEX
if redBallCenter is not None:
    cv2.circle(frame, redBallCenter, 5, (0, 0, 255), -1)
    red_ball_dist = measure_ball_distance(
        ball_center=redBallCenter,
        focal_length=320,  # Assuming the center of the ball is along the image's width
        camera_matrix=camera_matrix,
        pixel_size_in_cm=pixel_size_in_cm
    )
    cv2.putText(frame, f'Red: {red_ball_dist:.2f} cm', (redBallCenter[0] - 20, redBallCenter[1] - 20), font, 0.5, (0, 0, 255), 2)

if blueBallCenter is not None:
    cv2.circle(frame, blueBallCenter, 5, (255, 0, 0), -1)
    blue_ball_dist = measure_ball_distance(
        ball_center=blueBallCenter,
        focal_length=240,  # Assuming the center of the ball is along the image's height
        camera_matrix=camera_matrix,
        pixel_size_in_cm=pixel_size_in_cm
    )
    cv2.putText(frame, f'Blue: {blue_ball_dist:.2f} cm', (blueBallCenter[0] - 20, blueBallCenter[1] - 20), font, 0.5, (255, 0, 0), 2)

# Save the image with detected ball centroids and distances
output_folder = "distance_images"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "annotated_image.jpg")
cv2.imwrite(output_path, frame)

# Display the image with detected ball centroids and distances
cv2.imshow("Detected Balls and Distances", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
