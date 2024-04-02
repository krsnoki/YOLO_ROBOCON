import cv2
import numpy as np
# Load the trained SVM model
svm = cv2.ml.SVM_load('svm_model.xml')

# def detect_red_ball(image):
    # # Resize the image to the same size used for training
    # image = cv2.resize(image, (64, 64))

    # # Flatten the image
    # image = image.flatten()

    # cv2.imshow('image',image)
    # temp = cv2.circle(image, (32, 32), 30, (0, 0, 255), -1)
    # # cv2.imwrite("../output_frames",temp
    # cv2.imshow('image',temp)
    # # Use the SVM model to predict
    # cont, result = svm.predict(image.reshape(1, -1).astype(np.float32))

    # # Class 1 represents red ball, so if result is 1, it's a red ball

    # print(cont,result)
    # if result == 1:
    #     return True
    # else:
    #     return False

 
def detect_red_ball(image):
    resized_image = cv2.resize(image, (64, 64))

    resized_image = resized_image.flatten()

    cont, result = svm.predict(resized_image.reshape(1, -1).astype(np.float32))

    if result == 1:
        min_x, min_y, max_x, max_y = 0, 0, 0, 0

        for idx, row in enumerate(resized_image):
            if row == 255:
                if min_x == 0 or idx < min_x:
                    min_x = idx

                if max_x == 0 or idx > max_x:
                    max_x = idx

                if min_y == 0 or ((idx - max_x) // 64) < min_y:
                    min_y = (idx - max_x) // 64

                if max_y == 0 or ((idx - max_x) // 64) > max_y:
                    max_y = (idx - max_x) // 64

        center = (min_x + (max_x - min_x) // 2,
                  max_y + (max_y - min_y) // 2)

        annotated_image = cv2.circle(image, center, 0, (0, 255, 0), 3)

        cv2.imwrite("ball_Result.jpg", annotated_image)

        print("Red ball detected!")

        return True

    else:
        print("No red ball detected.")

        return False

test_image = cv2.imread('../Data/red_ball/rb27.jpg')
# Detect red ball in the test image
is_red_ball = detect_red_ball(test_image)

if is_red_ball == 1:
    print("Red ball detected!")
else:
    print("No red ball detected.")
