import shutil

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
