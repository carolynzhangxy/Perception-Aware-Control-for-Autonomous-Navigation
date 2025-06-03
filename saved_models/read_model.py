import tensorflow as tf

# Load the SavedModel
model = tf.saved_model.load('./saved_models/cnn_100kTrain')

# List available signatures, this helps understanding how to use the model
print(model.signatures)
