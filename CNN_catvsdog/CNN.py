import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 


def my_model():
    #Instantiate an empty model
    model = Sequential()
    
    # 1st Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='valid', activation='relu', input_shape=(100,100,1)))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=1, padding='valid'))
    
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='valid', activation='relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=1, padding='valid'))
    
    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # Fully Connected Layer
    model.add(Dense(64, activation='relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    

    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

    return model