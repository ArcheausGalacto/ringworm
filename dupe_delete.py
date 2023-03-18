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
