import cv2
import edgeiq
import os
import json
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

NOTE: This app will auto-detect and use an Intel Movidius Neural Compute stick if
present. However, not all models can make use of it!
"""

# Static keys for extracting data from config JSON file
CONFIG_FILE = 'alwaysai.app.json'
CLASSIFIERS = 'classifiers'
CLASSIFIERS_NEEDED_TO_AGREE = 'classifiers_needed_to_agree'
FOUND_FOLDER = 'detected_output_folder'
EMPTY_FOLDER = 'empty_output_folder'
MODEL_ID = 'model_id'
THRESHOLD = 'confidence_level_threshold'
TARGETS = 'target_labels'

# Static values
SOURCE_FOLDER = 'source_images'


def main():

    # Load configuration data from the alwaysai.app.json file
    config = load_json(
        CONFIG_FILE)
    classifiers_config = config[CLASSIFIERS]
    classifiers_needed_to_agree = config[CLASSIFIERS_NEEDED_TO_AGREE]
    found_folder = config[FOUND_FOLDER]
    empty_folder = config[EMPTY_FOLDER]

    # Spin up all the classifiers listed in the configuration file into an array
    classifiers = classifiers_from(classifiers_config, auto_detected_engine())

    # Get all paths for images in the designated source folder
    image_paths = sorted(list(edgeiq.list_images(SOURCE_FOLDER + '/')))

    # Info for console output
    starting_image_count = len(image_paths)
    print("app.py: main: Checking {} images from '{}' folder ...".format(
        starting_image_count, SOURCE_FOLDER))
    print('app.py: main: {:.1f}% or more of {} classifiers must be in agreement before a target object is considered found.'.format(
        classifiers_needed_to_agree * 100, len(classifiers)))

    for image_path in image_paths:
        image_display = cv2.imread(image_path)
        image = image_display.copy()

        # Run through all classifiers looking for target labels
        found = False
        if targets_detected_among(classifiers, image_path, image, classifiers_needed_to_agree) == True:
            found = True

        # Sort image to appropriate folder if target label detected
        sort_image_by_detection(found, image_path, empty_folder, found_folder)

    # Print info to console upon completion
    print("app.py: main: Completed sorting of {} images".format(starting_image_count))
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


def auto_detected_engine():
    '''
    Automatically use an Intel/Movidius Neural compute stick if available,
    otherwise run the onboard GPU. NOTE that not all models can make
    use of this USB device. If you get a `Failed to initialize Inference Engine backend`
    error, disconnect the stick and try running the app again.
    '''
    if is_accelerator_available() == True:
        print('app.py: auto_detected_engine: An accelerator was detected')
        return edgeiq.Engine.DNN_OPENVINO
    return edgeiq.Engine.DNN


def is_accelerator_available():
    """Detect if an Intel Neural Compute Stick accelerator is attached"""
    if edgeiq.find_usb_device(id_vendor=edgeiq.constants.NCS1_VID, id_product=edgeiq.constants.NCS1_PID) == True:
        return True
    if edgeiq.find_usb_device(edgeiq.constants.NCS1_VID, edgeiq.constants.NCS1_PID2) == True:
        return True
    if edgeiq.find_usb_device(edgeiq.constants.NCS2_VID, edgeiq.constants.NCS2_PID) == True:
        return True
    return False


def classifiers_from(array_of_objects, engine=edgeiq.Engine.DNN):
    '''
    Taking the configuration array of converted JSON objects and initializing all the classifiers. Returns an
    array of tuples (classifer, array_of_target_labels, confidence_level_threshold_for_detections)
    '''
    result = []
    for object in array_of_objects:
        model_id = object[MODEL_ID]
        confidence_level = object[THRESHOLD]
        targets = object[TARGETS]
        print('app.py: classifier_from: initializing classifier with model id: {}'.format(
            model_id))
        classifier = edgeiq.Classification(model_id)
        classifier.load(engine=engine)
        result.append((classifier, targets, confidence_level))
    return result


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


def targets_detected_among(classifiers_tuple_array, image_path, image, required_percent_of_models_in_agreement=0.5):
    '''
    Takes an array of tuples (classifier, array_of_target_labels, confidence_level) and checks to see if the minimum
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
    print('app.py: targets_detected_among: {:.1f}% of all models agree that a target object was found in {}'.format(
        percent_in_agreement*100, filename))
    if percent_in_agreement >= required_percent_of_models_in_agreement:
        return True
    return False


def targets_detected(classifier, filename, image, targets_array, target_confidence_level=0.6):
    '''
    Returns True if any prediction in the predictions list matches a label from the
    targets_array (an array of strings). Note that different classifiers may use
    different confidence level base numbers. Most will map 1.0 = 100%, but not all!
    '''
    results = classifier.classify_image(image, target_confidence_level)
    for prediction in results.predictions:
        for target in targets_array:
            if prediction.label == target:
                # UNCOMMENT if wanting to display confidence level values
                # print('app.py: targets_detected: {} with {} confidence from the {} model in {}'.format(
                # prediction.label, prediction.confidence, classifier.model_id, filename))
                return True
    return False


if __name__ == "__main__":
    main()
