
import numpy as np
import cv2
import os

np.random.seed(1)

def create_training(dir, categories, IMG_SIZE=100):
    
    training_data = []
    
    # saving images to an array
    for category in categories:

        class_label = categories.index(category)
        path = os.path.join(dir, category)

        # looping over the images in each category directory
        for img in os.listdir(path):
            try:
                image_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                # reshaping the arrays
                new_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))

                # appending the array to our training data file
                training_data.append([new_array, class_label])
            except Exception as e:
                pass
    
    # shuffle the training data
    np.random.shuffle(training_data)

    return training_data