# -*- coding: utf-8 -*-
"""Copy of convolutional_neural_network.ipynb

# Convolutional Neural Network

### Importing the libraries
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

tf.__version__

"""## Part 1 - Data Preprocessing"""

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',  # this is the target directory
        target_size=(64, 64),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

"""### Preprocessing the Training set"""

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

"""### Preprocessing the Test set

## Part 2 - Building the CNN

### Initialising the CNN
"""

cnn = tf.keras.models.Sequential()

"""### Step 1 - Convolution"""

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

"""### Step 2 - Pooling"""

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

"""### Adding a second convolutional layer"""

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

"""### Step 3 - Flattening"""

cnn.add(tf.keras.layers.Flatten())

"""### Step 4 - Full Connection"""

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

"""### Step 5 - Output Layer"""

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"""## Part 3 - Training the CNN

### Compiling the CNN
"""

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""### Training the CNN on the Training set and evaluating it on the Test set"""

cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

"""## Part 4 - Making a single prediction"""

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('sample/fake/fake', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'

print(prediction)
