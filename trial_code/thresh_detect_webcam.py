import cv2
import numpy as np

def nothing(x):
    pass

def detect_color(image, color_lower, color_upper, color_space='HSV'):
    if color_space == 'HSV':
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'GRAY':
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Color space not supported. Use 'HSV' or 'GRAY'.")

    mask = cv2.inRange(image_hsv, color_lower, color_upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result, mask

def main():
    # Create a VideoCapture object to read from webcam
    cap = cv2.VideoCapture(2)  # 0 for default webcam, change if using another camera

    # Create a window for trackbars
    cv2.namedWindow('Adjustment Bars')

    # Create trackbars for adjusting parameters
    cv2.createTrackbar('Hue Lower', 'Adjustment Bars', 0, 179, nothing)
    cv2.createTrackbar('Hue Upper', 'Adjustment Bars', 179, 179, nothing)
    cv2.createTrackbar('Saturation Lower', 'Adjustment Bars', 0, 255, nothing)
    cv2.createTrackbar('Saturation Upper', 'Adjustment Bars', 255, 255, nothing)
    cv2.createTrackbar('Value Lower', 'Adjustment Bars', 0, 255, nothing)
    cv2.createTrackbar('Value Upper', 'Adjustment Bars', 255, 255, nothing)
    cv2.createTrackbar('Grayscale Lower', 'Adjustment Bars', 0, 255, nothing)
    cv2.createTrackbar('Grayscale Upper', 'Adjustment Bars', 255, 255, nothing)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Get current trackbar positions
        hue_lower = cv2.getTrackbarPos('Hue Lower', 'Adjustment Bars')
        hue_upper = cv2.getTrackbarPos('Hue Upper', 'Adjustment Bars')
        sat_lower = cv2.getTrackbarPos('Saturation Lower', 'Adjustment Bars')
        sat_upper = cv2.getTrackbarPos('Saturation Upper', 'Adjustment Bars')
        val_lower = cv2.getTrackbarPos('Value Lower', 'Adjustment Bars')
        val_upper = cv2.getTrackbarPos('Value Upper', 'Adjustment Bars')
        gray_lower = cv2.getTrackbarPos('Grayscale Lower', 'Adjustment Bars')
        gray_upper = cv2.getTrackbarPos('Grayscale Upper', 'Adjustment Bars')

        # Define the lower and upper HSV values for Red, Blue, and Purple
        red_lower = np.array([0, sat_lower, val_lower])
        red_upper = np.array([hue_upper, sat_upper, val_upper])

        blue_lower = np.array([hue_lower, sat_lower, val_lower])
        blue_upper = np.array([hue_upper, sat_upper, val_upper])

        purple_lower = np.array([130, sat_lower, val_lower])
        purple_upper = np.array([160, sat_upper, val_upper])

        # Threshold to detect red color in HSV
        red_result, red_mask = detect_color(frame, red_lower, red_upper, color_space='HSV')

        # Threshold to detect blue color in HSV
        blue_result, blue_mask = detect_color(frame, blue_lower, blue_upper, color_space='HSV')

        # Threshold to detect purple color in HSV
        purple_result, purple_mask = detect_color(frame, purple_lower, purple_upper, color_space='HSV')

        # Threshold to detect color in grayscale
        gray_lower = np.array([gray_lower])
        gray_upper = np.array([gray_upper])
        gray_result, gray_mask = detect_color(frame, gray_lower, gray_upper, color_space='GRAY')

        # Display the results
        cv2.imshow('Original Image', frame)
        cv2.imshow('Red Detected', red_result)
        cv2.imshow('Red Mask', red_mask)
        cv2.imshow('Blue Detected', blue_result)
        cv2.imshow('Blue Mask', blue_mask)
        cv2.imshow('Purple Detected', purple_result)
        cv2.imshow('Purple Mask', purple_mask)
        cv2.imshow('Grayscale Detected', gray_result)
        cv2.imshow('Grayscale Mask', gray_mask)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
