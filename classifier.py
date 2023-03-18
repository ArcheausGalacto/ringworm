import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
            if "NotRingworm" in folder:
                labels.append(0)
            else:
                labels.append(1)
    return images, labels

def create_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    not_ringworm_folder = r"C:/Users/Danie/OneDrive/Desktop/ringworm_classifier_V2/data/NotRingworm"
    ringworm_folder = r"C:/Users/Danie/OneDrive/Desktop/ringworm_classifier_V2/data/Ringworm"

    not_ringworm_images, not_ringworm_labels = load_images(not_ringworm_folder)
    ringworm_images, ringworm_labels = load_images(ringworm_folder)

    images = not_ringworm_images + ringworm_images
    labels = not_ringworm_labels + ringworm_labels

    X_train, X_test, y_train, y_test = train_test_split(np.array(images), np.array(labels), test_size=0.2, random_state=42)

    datagen = ImageDataGenerator(rescale=1.0/255)
    train_generator = datagen.flow(X_train, y_train, batch_size=32)

    model = create_model()
    model.fit(train_generator, epochs=5)

    X_test = X_test.astype("float32") / 255
    y_pred = np.argmax(model.predict(X_test), axis=-1)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("Confusion Matrix:\n", cm)
    print("Accuracy:", accuracy)
