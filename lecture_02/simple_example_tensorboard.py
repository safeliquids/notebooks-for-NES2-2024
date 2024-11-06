
# The first example - simple Keras MLP model with TensorBoard to 
# monitor the training process in real time

# Steps to use TensorBoard with this script:
# 1. Run this script to train the model and generate logs for TensorBoard.
# 2. In your terminal or shell, start TensorBoard by running:
#    tensorboard --logdir logs/fit --reload_interval=5
# 3. Open your web browser and go to the URL: http://localhost:6006/ 
#    to view real-time training metrics (e.g., loss, accuracy) and other visualizations.
#
# The model is trained on the MNIST dataset, reshaped for use in a fully connected 
# neural network. The logs for TensorBoard are saved in the 'logs/fit/' directory 
# with a timestamp to organize different training runs.

import keras
from keras.datasets import mnist
import os
import datetime

###############################################
# Define the log directory for TensorBoard
log_dir = "./logs/fit/simple_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# print(os.getcwd()) 

# Clear any logs from previous runs
# !rm -rf ./logs/

# Ensure the directory exists
if not os.path.exists(os.path.dirname(log_dir)):
    os.makedirs(os.path.dirname(log_dir))

###############################################
# Initialize Tensorboard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1,  # Visualize histograms of layer weights
    write_graph=True,  # Log the graph to visualize the model structure
    write_images=True  # Optionally, save images of weights and activation histograms
    # update_freq='batch'  # Log metrics after every batch
    # write_steps_per_second=True  # Log steps per second during training
)

###############################################
# Load (and analyze) the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pre-process the data
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

###############################################
# Define the model
model = keras.Sequential([
    keras.layers.InputLayer(shape=(28 * 28,)),
    keras.layers.Dense(256, activation='relu'),
    #  keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Summarize the model
model.summary()

# Set model parameters
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

###############################################
# Train the model with the Tensorboard callback
num_epochs = 8
batch_size = 32

# Train the model with callbacks
model.fit(x_train, y_train, epochs=num_epochs, batch_size = batch_size, callbacks=[tensorboard_callback])

###############################################
# Evaluate the model
test_logs = model.evaluate(x_test, y_test)

# Save the model
#model.save('mnist_simple_mlp_model_0.keras')

# Load the model
# loaded_model = keras.models.load_model('mnist_simple_mlp_model_0.keras')