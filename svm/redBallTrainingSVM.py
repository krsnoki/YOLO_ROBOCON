import cv2
import numpy as np
import os

# Function to load images and label them
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(label)
    return images, labels

# Load red ball images and label them as 1
red_balls, red_labels = load_images_from_folder('../Data/red_ball', 1)


# Load non-red ball images and label them as 0
white_balls, white_labels = load_images_from_folder('../Data/white_ball', 0)
print(white_balls)
# Combine red and non-red ball images and labels
images = red_balls + white_balls
print("images: ", images)
labels = red_labels + white_labels
print("Red labels: ", red_labels)
# Convert images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Flatten the images
images = images.reshape((images.shape[0], -1))

# Shuffle the data
shuffle_index = np.random.permutation(len(images))
images = images[shuffle_index]
labels = labels[shuffle_index]

# Create SVM model
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(1)

# Train the SVM
svm.train(images.astype(np.float32), cv2.ml.ROW_SAMPLE, labels)

# Save the SVM model
svm.save('svm_model.xml')
