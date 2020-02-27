import cv2
import edgeiq
import filters
import os
import shutil

"""
Use image classification to classify a batch of images. The
classification labels can be changed by selecting different models.
Different images can be used by updating the files in the *images/*
directory. Note that when developing for a remote device, removing
images in the local *images/* directory won't remove images from the
device. They can be removed using the `aai app shell` command and
deleting them from the *images/* directory on the remote device.

To change the computer vision model, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html
"""
ANIMALS_FOLDER = '/output_images/animals'
NO_ANIMALS_FOLDER = '/output_images/no_animals'


def main():
    classifier = edgeiq.Classification("alwaysai/googlenet")
    classifier.load(engine=edgeiq.Engine.DNN)

    print("Engine: {}".format(classifier.engine))
    print("Accelerator: {}\n".format(classifier.accelerator))
    print("Model:\n{}\n".format(classifier.model_id))
    print("Labels:\n{}\n".format(classifier.labels))

    image_paths = sorted(list(edgeiq.list_images("source_images/")))
    print("Images:\n{}\n".format(image_paths))

    for image_path in image_paths:
        image_display = cv2.imread(image_path)
        image = image_display.copy()

        confidence_level = 0.3
        predictions = classifier.classify_image(image, confidence_level)
        results = edgeiq.filter_predictions_by_label(
            predictions, filters.shufflenet())

        path, filename = os.path.split(image_path)
        # path = path/to/file
        # filename = foobar.txt

        # Default move file to no_animals folder
        new_path = path + NO_ANIMALS_FOLDER + filename

        if results.predictions:
            # Move file to animals folder if any animal detected
            new_path = path + ANIMALS_FOLDER + filename

        shutil.move(image_path, new_path)

    print("End of Line")


if __name__ == "__main__":
    main()
