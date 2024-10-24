import tensorflow as tf

# Load the .h5 model
model = tf.keras.models.load_model('final_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to a .tflite file
with open('fire_detection.tflite', 'wb') as f:
    f.write(tflite_model)
