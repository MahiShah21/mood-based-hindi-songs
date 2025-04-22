from tensorflow import keras

# Load the legacy HDF5 model
old_model = keras.models.load_model("C:/Users/dell/Downloads/tensorflow.h5", compile=False)

# Save it in the new .keras format
old_model.save("C:/Users/dell/Downloads/tensorflow.keras", save_format="keras")

print("Model converted and saved as tensorflow.keras")
