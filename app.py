import cv2
import edgeiq
import filters
import os
import shutil

"""
Use image classification to sort a batch of images. The
classification labels can be changed by selecting different models.
Different images can be used by updating the files in the *source_images/*
directory. Note that when developing for a remote device, removing
images in the local *images/* directory won't remove images from the
device. They can be removed using the `aai app shell` command and
deleting them from the *images/* directory on the remote device.

To change the computer vision model, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html
"""
ANIMALS_FOLDER = 'output_images/animals/'
NO_ANIMALS_FOLDER = 'output_images/no_animals/'


def main():
    classifier = edgeiq.Classification("alwaysai/squeezenet_v1.1")
    classifier.load(engine=edgeiq.Engine.DNN)

    print("Engine: {}".format(classifier.engine))
    print("Accelerator: {}\n".format(classifier.accelerator))
    print("Model:\n{}\n".format(classifier.model_id))
    print("Labels:\n{}\n".format(classifier.labels))

    image_paths = sorted(list(edgeiq.list_images("source_images/")))
    starting_image_count = len(image_paths)
    print("Checking {} images".format(starting_image_count))

    for image_path in image_paths:
        image_display = cv2.imread(image_path)
        image = image_display.copy()

        # Set confidence threshold for animal detection
        confidence_level = 0.2
        results = classifier.classify_image(image, confidence_level)

        # Get filepath and name of current file from image_path
        path, filename = os.path.split(image_path)

        # Set the target labels interested in
        filter = filters.lsvrc_animals()

        # Determine which folder to sort file to dependent on labels detected
        new_path = output_path(
            path, filename, results.predictions, filter, confidence_level)

        # Move file to appropriate output folder
        shutil.move(image_path, new_path)

    animal_images_count = len(
        list(edgeiq.list_images("output_images/animals")))

    print("Sorting of {} images complete".format(starting_image_count))
    print("{} images with animals detected".format(animal_images_count))


def output_path(original_image_path, filename, predictions, filter, confidence):
    no_animal_path = original_image_path.replace(
        'source_images', NO_ANIMALS_FOLDER)
    animal_path = original_image_path.replace(
        'source_images', ANIMALS_FOLDER)
    for prediction in predictions:
        for animal in filter:
            if prediction.label == animal:
                print('app.py: output_path: animal detected: {} with {} confidence in file {}. Returning path: {}'.format(
                    prediction.label, prediction.confidence, filename, animal_path))
                return animal_path
    return no_animal_path


if __name__ == "__main__":
    main()
