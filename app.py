import cv2
import edgeiq
import os
import json
import shutil

"""
This is a simplified version off app.py. Support for multiple classifiers and auto detection
of an Intel Movidius accelerator stick removed.

Use image classification to sort a batch of images. The
classification labels can be changed by selecting different models.
Different images can be used by updating the files in the *source_images/*
directory. Note that when developing for a remote device, removing
images in the local *images/* directory won't remove images from the
device. They can be removed using the `aai app shell` command and
deleting them from the *images/* directory on the remote device.

To change the computer vision model, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

NOTE: This app will auto-detect and use an Intel Movidius Neural Compute stick if
present. However, not all models can make use of it!
"""

# Static keys for extracting data from config JSON file
CONFIG_FILE = 'alwaysai.app.json'
CLASSIFIERS = 'classifiers'
FOUND_FOLDER = 'found_folder'
EMPTY_FOLDER = 'empty_folder'
MODEL_ID = 'model_id'
THRESHOLD = 'minimum_confidence_level'
TARGETS = 'target_labels'

# Static values
SOURCE_FOLDER = 'source_images'


def main():

    # 1. Load configuration data from the alwaysai.app.json file
    config = load_json(
        CONFIG_FILE)
    classifier_config = config[CLASSIFIERS][0]
    found_folder = config[FOUND_FOLDER]
    empty_folder = config[EMPTY_FOLDER]

    # 2. Spin up just the first classifier listed in the configuration file
    model_id = classifier_config[MODEL_ID]
    confidence_level = classifier_config[THRESHOLD]
    targets = classifier_config[TARGETS]
    print('app.py: classifier_from: initializing classifier with model id: {}'.format(
        model_id))
    classifier = edgeiq.Classification(model_id)
    classifier.load(engine=edgeiq.Engine.DNN)

    # 3. Loop through all source images
    image_paths = sorted(list(edgeiq.list_images(SOURCE_FOLDER + '/')))
    # Info for console output
    image_count = len(image_paths)
    print("app.py: main: Checking {} images from '{}' folder ...".format(
        image_count, SOURCE_FOLDER))
    for image_path in image_paths:
        image_display = cv2.imread(image_path)
        image = image_display.copy()

        found = False

        # 4-5. Find a label match between detected objects and listed target
        results = classifier.classify_image(image, confidence_level)
        for prediction in results.predictions:
            for target in targets:
                if prediction.label == target:
                    # A target label was found among the detected images
                    _, filename = os.path.split(image_path)
                    print('app.py: main: {} found with {} confidence from the {} model in {}'.format(
                        prediction.label, prediction.confidence, classifier.model_id, filename))
                    found = True

        # 6. Sort image to appropriate folder if target label detected
        sort_image_by_detection(found, image_path, empty_folder, found_folder)

    # Print info to console upon completion
    print("app.py: main: Completed sorting of {} images".format(image_count))
    found_images_count = len(
        list(edgeiq.list_images(found_folder)))
    print("app.py: main: {} images in the output folder".format(
        found_images_count))


def load_json(filepath):
    '''
    Convenience to check and load a JSON file
    '''
    if os.path.exists(filepath) == False:
        raise Exception(
            'app.py: load_json: File at {} does not exist'.format(filepath))
    with open(filepath) as data:
        return json.load(data)


def sort_image_by_detection(did_detect, image_path, empty_folder, found_folder):
    '''
    Sort images to appropriate folders.
    '''
    # Determine which folder to sort file to dependent on labels detected
    new_path = image_path.replace(
        SOURCE_FOLDER, empty_folder)
    if did_detect == True:
        new_path = image_path.replace(SOURCE_FOLDER, found_folder)

    # Move file to appropriate output folder
    shutil.move(image_path, new_path)


if __name__ == "__main__":
    main()
