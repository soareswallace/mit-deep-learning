import os
from tensorflow import keras
from pre_process import preprocess_images
from plot_images import plot_images
from model_build import model_builder
import matplotlib.pyplot as plt

this_repo_url = 'https://github.com/lexfridman/mit-deep-learning/raw/master/'
this_tutorial_url = os.path.join(this_repo_url, 'tutorial_deep_learning_basics')


(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

# plot_images(train_images, train_labels)

model = model_builder(28, 28)
history = model.fit(train_images, train_labels, epochs=5)

print(test_images.shape)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test Accuracy: ", test_acc)

