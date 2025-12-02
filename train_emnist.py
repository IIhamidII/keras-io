"""
Title: Simplified Letters + Digits Classifier
Description: Train a model to recognize digits (0-9) and letters (A-E)
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageDraw, ImageFont
import random

print("Creating simplified dataset (0-9, A-E)...")

# Load MNIST for digits
from tensorflow.keras.datasets import mnist
(x_mnist, y_mnist), (x_mnist_test, y_mnist_test) = mnist.load_data()

# Generate synthetic letter data (A-E)
def generate_letter_samples(letter, num_samples=6000):
    """Generate synthetic handwritten-style letters"""
    samples = []
    
    # Map letters to class indices (A=10, B=11, C=12, D=13, E=14)
    letter_to_idx = {'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14}
    
    for _ in range(num_samples):
        # Create blank image
        img = Image.new('L', (28, 28), color=0)
        draw = ImageDraw.Draw(img)
        
        # Random variations for handwriting simulation
        font_size = random.randint(16, 26)
        x_offset = random.randint(1, 10)
        y_offset = random.randint(1, 10)
        
        # Draw letter (using default font)
        try:
            # Try to use a system font
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()
        
        draw.text((x_offset, y_offset), letter, fill=255, font=font)
        
        # Convert to array for augmentation
        img_array = np.array(img)
        
        # Add more aggressive augmentations for better variation
        # 1. Random rotation
        angle = random.uniform(-15, 15)
        img = Image.fromarray(img_array)
        img = img.rotate(angle, fillcolor=0)
        img_array = np.array(img)
        
        # 2. Random scaling/stretching
        if random.random() > 0.5:
            scale_x = random.uniform(0.85, 1.15)
            scale_y = random.uniform(0.85, 1.15)
            new_width = int(28 * scale_x)
            new_height = int(28 * scale_y)
            img = Image.fromarray(img_array)
            img = img.resize((new_width, new_height))
            # Paste back onto 28x28 canvas
            canvas = Image.new('L', (28, 28), color=0)
            offset_x = (28 - new_width) // 2
            offset_y = (28 - new_height) // 2
            canvas.paste(img, (offset_x, offset_y))
            img_array = np.array(canvas)
        
        # 3. Add noise for variation
        noise = np.random.normal(0, 15, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # 4. Random brightness adjustment
        brightness_factor = random.uniform(0.7, 1.3)
        img_array = np.clip(img_array * brightness_factor, 0, 255).astype(np.uint8)
        
        samples.append(img_array)
    
    return np.array(samples)

print("Generating letter samples...")
letters_data = []
letters_labels = []

for idx, letter in enumerate(['A', 'B', 'C', 'D', 'E']):
    print(f"  Generating {letter}...")
    # Generate more samples for B to improve recognition
    num_samples = 12000 if letter == 'B' else 6000
    samples = generate_letter_samples(letter, num_samples=num_samples)
    letters_data.append(samples)
    letters_labels.extend([10 + idx] * len(samples))

letters_data = np.concatenate(letters_data, axis=0)
letters_labels = np.array(letters_labels)

# Combine MNIST digits with letters
x_train = np.concatenate([x_mnist, letters_data], axis=0)
y_train = np.concatenate([y_mnist, letters_labels], axis=0)

# Create test set (smaller)
letters_test_data = []
letters_test_labels = []
for idx, letter in enumerate(['A', 'B', 'C', 'D', 'E']):
    samples = generate_letter_samples(letter, num_samples=1000)
    letters_test_data.append(samples)
    letters_test_labels.extend([10 + idx] * len(samples))

letters_test_data = np.concatenate(letters_test_data, axis=0)
letters_test_labels = np.array(letters_test_labels)

x_test = np.concatenate([x_mnist_test, letters_test_data], axis=0)
y_test = np.concatenate([y_mnist_test, letters_test_labels], axis=0)

# Shuffle
train_indices = np.random.permutation(len(x_train))
x_train = x_train[train_indices]
y_train = y_train[train_indices]

test_indices = np.random.permutation(len(x_test))
x_test = x_test[test_indices]
y_test = y_test[test_indices]

print(f"\nDataset created:")
print(f"  Classes: 15 (0-9, A-E)")
print(f"  Train samples: {len(x_train)}")
print(f"  Test samples: {len(x_test)}")

# Model parameters
num_classes = 15  # 0-9 + A-E
input_shape = (28, 28, 1)

# Preprocess
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build model
model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax"),
])

model.summary()

# Train
batch_size = 128
epochs = 5

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("\nTraining model...")
model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)

# Evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

# Save
model.save('letters_digits_model.h5')
print("Model saved to letters_digits_model.h5")

