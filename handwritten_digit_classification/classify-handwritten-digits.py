# import keras
import tensorflow as tf

# Read the MNIST data form keras collection of datasets
data = tf.keras.datasets.mnist
# Split the data
(x_train, y_train), (x_test, y_test) = data.load_data()
# The MNIST data contains two-dimensional images. Flatten them into one-dimensional images for convenience
x_train, x_test = x_train/255.0, x_test/255.0

number_of_units = 10  # The hidden layer contains 10 neurons

# Create the neural network: one hidden layer with 10 neurons
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
model.evaluate(x_test, y_test)
