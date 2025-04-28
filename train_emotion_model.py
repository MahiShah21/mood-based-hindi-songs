import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size and batch size
img_height, img_width = 48, 48
batch_size = 64
epochs = 30

# Define the directory where the "train" and "test" folders are located
data_dir = 'C:\Users\dell\Desktop\MusicApp\mood-based-hindi-songs\archive'  # Replace with the actual path to the "archive" folder

# Create ImageDataGenerator for training data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of training data for validation
)

# Create ImageDataGenerator for test data (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data from the "train" directory
train_generator = train_datagen.flow_from_directory(
    data_dir + '/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'  # Specify training subset
)

# Load validation data from the "train" directory
validation_generator = train_datagen.flow_from_directory(
    data_dir + '/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'  # Specify validation subset
)

# Load test data from the "test" directory
test_generator = test_datagen.flow_from_directory(
    data_dir + '/test',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical'
)

# Define the model (same as before)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 output classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model using the generators
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Evaluate the model on the test generator
loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size, verbose=0)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save the trained model
model.save('emotion_recognition_model.h5')
print("Trained model saved as emotion_recognition_model.h5")