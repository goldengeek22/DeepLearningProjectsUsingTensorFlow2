import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

# Declare parameters
Number_of_features = 2
Number_of_units = 1  # indicates number of neurons

# Declare the Weights and Bias
weights = tf.Variable(tf.zeros([Number_of_features, Number_of_units]))  # initializing to zero
bias = tf.Variable(tf.zeros([Number_of_units]))  # Initializing the bias to zero


# Define the Perceptron Function
def perceptron(x):
    I = tf.add(tf.matmul(x, weights), bias)  # Computes the matrix multiplication of x and weight
    output = tf.sigmoid(I)  # Calculates the output using the Sigmoid activation function
    return output


optimizer = tf.keras.optimizers.Adam(.01)

# Read in the Data
df = pd.read_csv('data/training_data.csv')
df.head()

# Visualization of labels
# plt.scatter(df.x1, df.x2, c=df.label)

# Preparing Inputs
x_input = df[['x1', 'x2']].to_numpy()
y_label = df[['label']].to_numpy()

# Initialize variables
x = tf.Variable(x_input)
x = tf.cast(x, tf.float32)
y = tf.Variable(y_label)
y = tf.cast(y, tf.float32)


# Define the Loss Function and Optimizer
def individual_loss():
    return abs(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=perceptron(x))))


# Train the model
for i in range(1000):
    optimizer.minimize(individual_loss, [weights, bias])

# New Values for Weights and Bias
tf.print(weights)
tf.print(bias)

# View the final loss
final_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=perceptron(x)))
tf.print(final_loss)

# Predicting Using the Trained Model
prediction_y = perceptron(x)
prediction_y_round = tf.round(prediction_y)  # Round off the output value to 1 or 0, to make the comparison with the target easier
tf.print(prediction_y_round)

# Evaluate the Model
print(accuracy_score(y, prediction_y_round))

# Generate the confusion matrix
confusion_matrix(y, prediction_y_round)
