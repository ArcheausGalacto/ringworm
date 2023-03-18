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
scrape_images("ringworm", "C:/Users/Danie/OneDrive/Desktop/ringworm_classifier_V2/untested", 1000)
