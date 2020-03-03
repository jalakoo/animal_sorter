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

USE_MOVIDIUS_ACCELERATOR = False
ANIMALS_FOLDER = 'output_images/animals/'
NO_ANIMALS_FOLDER = 'output_images/no_animals/'
TARGETS = filters.lsvrc_animals()
CLASSIFIERS = [('alwaysai/squeezenet_v1.1', TARGETS, 0.2),
               ('alwaysai/shufflenet_1x_g3', TARGETS, 9.0),
               ('alwaysai/inception_resnet', TARGETS, 0.2)]
CLASSIFIERS_NEEDED_TO_AGREE = .5


def main():
    # Using an accelerator?
    engine = edgeiq.Engine.DNN
    if USE_MOVIDIUS_ACCELERATOR == True:
        engine = edgeiq.Engine.DNN_OPENVINO

    # Spin up all the classifiers you want to use
    classifiers = classifiers_from(CLASSIFIERS, engine)

    image_paths = sorted(list(edgeiq.list_images("source_images/")))
    starting_image_count = len(image_paths)
    print("Checking {} images".format(starting_image_count))

    for image_path in image_paths:
        image_display = cv2.imread(image_path)
        image = image_display.copy()

        # Run through all classifiers looking for target labels
        animal_found = False
        if targets_detected_among(classifiers, image_path, image, CLASSIFIERS_NEEDED_TO_AGREE) == True:
            animal_found = True

        # Determine which folder to sort file to dependent on labels detected
        new_path = image_path.replace(
            'source_images', NO_ANIMALS_FOLDER)
        if animal_found == True:
            new_path = image_path.replace('source_images', ANIMALS_FOLDER)

        # Move file to appropriate output folder
        shutil.move(image_path, new_path)

    animal_images_count = len(
        list(edgeiq.list_images("output_images/animals")))

    print("Sorting of {} images complete".format(starting_image_count))
    print("{} images with animals detected in output folder".format(
        animal_images_count))


def classifiers_from(an_array_of_tuples, engine=edgeiq.Engine.DNN):
    '''
    Taking the configuration array and initializing all the classifiers. Returns an
    array of tuples (classifer, array_of_target_labels, confidence_level_threshold_for_detections)
    '''
    result = []
    for model_id, targets, confidence_level in an_array_of_tuples:
        print('app.py: classifier_from: initializing classifier with model id: {}'.format(
            model_id))
        classifier = edgeiq.Classification(model_id)
        classifier.load(engine=engine)
        result.append((classifier, targets, confidence_level))
    return result


def targets_detected_among(classifiers_tuple_array, image_path, image, required_percent_of_models_in_agreement=0.5):
    '''
    Takes an array of tuples (classifier, confidence_level) and checks to see if the minimum
    percent of all available classifiers agree that the target array of labels is present.
    ie if percentage_of_models_in_agreement = 0.5 then at least half of the available models
    must return True for this function to return True. 1.0 would mean all available models
    would have to return True.
    '''
    _, filename = os.path.split(image_path)
    result = []
    for classifier, targets, confidence_level in classifiers_tuple_array:
        were_targets_detected = targets_detected(
            classifier, filename, image, targets, confidence_level)
        result.append(were_targets_detected)
    percent_in_agreement = sum(result) / len(result)
    print('app.py: targets_detected_among: {} of all models are in agreement that a target object was detected in file {}'.format(
        percent_in_agreement, filename))
    if percent_in_agreement >= required_percent_of_models_in_agreement:
        return True
    return False


def targets_detected(classifier, filename, image, targets_array, target_confidence_level=0.6):
    '''
    Returns True if any prediction in the predictions list matches a label from the
    targets_array (an array of strings). Note that different classifiers may use
    different confidence level base numbers. Most will map 1.0 = 100%, but not all!
    '''
    # _, filename = os.path.split(image_path)
    results = classifier.classify_image(image, target_confidence_level)
    for prediction in results.predictions:
        for target in targets_array:
            if prediction.label == target:
                print('app.py: targets_detected: {} with {} confidence from the {} model in {}'.format(
                    prediction.label, prediction.confidence, classifier.model_id, filename))
                return True
    return False


if __name__ == "__main__":
    main()
