import pandas as pd
import tensorflow as tf

# Read data
products_subscriptions_df = pd.read_csv('data/subscription.csv')
features_names = ['profession', 'salary', 'married', 'genre', 'vehicle', 'own_house', 'children', 'age']
features_input = products_subscriptions_df[features_names].to_numpy()
target_labels = products_subscriptions_df['product'].to_numpy()
print(target_labels)
# Initialize variables
x = tf.constant(features_input)
y = tf.constant(target_labels)

# Declare parameters
number_of_features = len(features_names)
number_of_units = 1  # Number of perceptron

# Declare the weights and bias
weights = tf.Variable(tf.zeros([number_of_features, number_of_units]))
bias = tf.Variable(tf.zeros([number_of_units]))


# Define the perceptron function
def perceptron(training_example):
    net_input = tf.add(tf.matmul(training_example, weights), bias)  # Computes the matrix multiplication of training example and weights
    output = tf.sigmoid(net_input)  # Calculates the output using the Sigmoid Activation function
    return output


optimizer = tf.keras.optimizers.Adam(.01)


# Define the loss function
def individual_loss():
    return abs(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=perceptron(x))))


# Train the model
for i in range(1000):
    optimizer.minimize(individual_loss, [weights, bias])

# New Values for Weights and Bias
tf.print(weights)
tf.print(bias)
