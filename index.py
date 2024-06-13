import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_images(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels, class_names

data_dir = 'archive/fruits-360_dataset/fruits-360/Training'
images, labels, class_names = load_and_preprocess_images(data_dir)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print('Passo 1 concluído!\n')

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=10, validation_data=validation_generator)

def classify_image(img_path, model, class_names, img_size=(64, 64)):
    img = cv2.imread(img_path)
    
    if img is None:
        raise ValueError(f"Error: Unable to load image at path {img_path}")

    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    class_name = class_names[class_idx]
    confidence = predictions[0][class_idx]
    
    return class_name, confidence

test_img_path = './strawberry.jpg'
if not os.path.exists(test_img_path):
    raise FileNotFoundError(f"O caminho da imagem de teste {test_img_path} não existe.")

class_name, confidence = classify_image(test_img_path, model, class_names)

print(f"Class: {class_name}, Confidence: {confidence:.2f}")


test_img_path = './kiwi.jpg'
if not os.path.exists(test_img_path):
    raise FileNotFoundError(f"O caminho da imagem de teste {test_img_path} não existe.")

class_name, confidence = classify_image(test_img_path, model, class_names)

print(f"Class: {class_name}, Confidence: {confidence:.2f}")

test_img_path = './limes.jpg'
if not os.path.exists(test_img_path):
    raise FileNotFoundError(f"O caminho da imagem de teste {test_img_path} não existe.")

class_name, confidence = classify_image(test_img_path, model, class_names)

print(f"Class: {class_name}, Confidence: {confidence:.2f}")

test_img_path = './lychee.jpg'
if not os.path.exists(test_img_path):
    raise FileNotFoundError(f"O caminho da imagem de teste {test_img_path} não existe.")

class_name, confidence = classify_image(test_img_path, model, class_names)

print(f"Class: {class_name}, Confidence: {confidence:.2f}")

test_img_path = './mango.jpg'
if not os.path.exists(test_img_path):
    raise FileNotFoundError(f"O caminho da imagem de teste {test_img_path} não existe.")

class_name, confidence = classify_image(test_img_path, model, class_names)

print(f"Class: {class_name}, Confidence: {confidence:.2f}")