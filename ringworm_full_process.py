import os
import requests
import random
import string
import time
from bs4 import BeautifulSoup

def scrape_images(query, folder_name, num_images):
    # Make the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Scrape images from random pages of search results
    for i in range(num_images):
        # Randomly select a page number
        start = random.randint(0, 1000)
        
        # Search for images on the selected page
        url = "https://www.google.com/search?q=" + query + "&tbm=isch&start=" + str(start)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        images = soup.find_all("img")
        
        # Shuffle the images
        random.shuffle(images)
        
        # Save the first image in the folder
        image_url = images[0]["src"]
        if not image_url.startswith("http") or "googlelogo" in image_url:
            continue
        if not image_url.startswith("http"):
            image_url = "https://www.google.com" + image_url
        response = requests.get(image_url)
        if response.status_code == 200:
            random_file_name = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
            with open(f"{folder_name}/{random_file_name}.jpg", "wb") as f:
                f.write(response.content)
        time.sleep(random.uniform(0.1, 0.2))

# Example usage
scrape_images("ringworm", "C:/Users/Danie/OneDrive/Desktop/ringworm_classifier_V2/untested", 100)


import shutil

import cv2
import numpy as np

def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img_resized = cv2.resize(img, (224, 224))
            images.append(img_resized)
            labels.append(1 if "Ringworm" in folder else 0)
    return images, labels

def load_and_classify_images(model, folder):
    images = []
    filepaths = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img_resized = cv2.resize(img, (224, 224))
            images.append(img_resized)
            filepaths.append(os.path.join(folder, filename))

    images = np.array(images).astype("float32") / 255
    predictions = np.argmax(model.predict(images), axis=-1)

    return filepaths, predictions

def move_classified_images(filepaths, predictions, target_folder):
    for filepath, prediction in zip(filepaths, predictions):
        if prediction == 1:
            dest = os.path.join(target_folder, os.path.basename(filepath))
            shutil.move(filepath, dest.replace("\\", "/"))

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

    model.fit(train_generator, epochs=5)

    # Load and classify untested images
    untested_folder = r"C:\Users\Danie\OneDrive\Desktop\ringworm_classifier_V2\untested"
    tested_folder = r"C:\Users\Danie\OneDrive\Desktop\ringworm_classifier_V2\tested"

    filepaths, predictions = load_and_classify_images(model, untested_folder)

    # Move classified ringworm images to the tested folder
    move_classified_images(filepaths, predictions, tested_folder)
    
import os
import shutil

def move_files(src_folder, dest_folder):
    for filename in os.listdir(src_folder):
        src = os.path.join(src_folder, filename)
        dest = os.path.join(dest_folder, filename)
        shutil.move(src, dest.replace("\\", "/"))

if __name__ == "__main__":
    tested_folder = r"C:\Users\Danie\OneDrive\Desktop\ringworm_classifier_V2\tested"
    ringworm_folder = r"C:\Users\Danie\OneDrive\Desktop\ringworm_classifier_V2\data\Ringworm"

    untested_folder = r"C:\Users\Danie\OneDrive\Desktop\ringworm_classifier_V2\untested"
    not_ringworm_folder = r"C:\Users\Danie\OneDrive\Desktop\ringworm_classifier_V2\data\NotRingworm"

    move_files(tested_folder, ringworm_folder)
    move_files(untested_folder, not_ringworm_folder)

import os
import sys
import hashlib
from PIL import Image

def md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def delete_duplicate_images(directory):
    if not os.path.exists(directory):
        print("Directory does not exist.")
        return

    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    hashes = {}

    for image_file in image_files:
        file_path = os.path.join(directory, image_file)
        file_md5 = md5(file_path)

        if file_md5 in hashes:
            os.remove(file_path)
            print(f"Removed duplicate image: {file_path}")
        else:
            hashes[file_md5] = file_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_duplicate_images.py <directory>")
        sys.exit(1)

    directory = "C:/Users/Danie/OneDrive/Desktop/ringworm_classifier_V2/data/NotRingworm"
    delete_duplicate_images(directory)
    directory = "C:/Users/Danie/OneDrive/Desktop/ringworm_classifier_V2/data/Ringworm"
    delete_duplicate_images(directory)
