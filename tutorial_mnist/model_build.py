import tensorflow as tf
from tensorflow import keras

def model_builder(input_x, input_y):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(input_x, input_y)), # flatten image to a 1d-array of 28 * 28 = 784 pixels
        keras.layers.Dense(128, activation=tf.nn.relu), # fully-connected layer (128 nodes)
        keras.layers.Dense(10, activation=tf.nn.softmax) # softmax layer returns an array of 10 probability scores that sum to 1
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'])

    return model
    


