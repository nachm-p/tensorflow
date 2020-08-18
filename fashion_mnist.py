'''
70k images
10 categories
images are 28x28
https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab2-Computer-Vision.ipynb
'''

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# normalisation between 0 a 1
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.imshow(train_images[0])
print(train_labels[0])
print(train_images[0])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    # 28 x 28 size of image
    # flatten -> making a 1d set from 2d
    keras.layers.Dense(128, activation=tf.nn.relu),
    # 128 functions
    # relu -> for x>0: x else 0
    keras.layers.Dense(10, activation=tf.nn.softmax)
    # softmax -> largest 1 and rest 0
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels)

#test_loss, tes_acc = model.evaluate(test_images, test_labels)

#predictions = model.predict(my_images)