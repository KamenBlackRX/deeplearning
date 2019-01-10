import math

import cv2
import numpy as np
from matplotlib import pyplot as plt


def _find_single_centroid(input_image: np.ndarray):
    """
    Find the centroid of a single mask.

    Arguments:
        :param input_image: image with mask information
        :type input_image: np.ndarray
    Returns:
        :return: tuple with the centroid information.
    """

    input_image[input_image > 0] = 255
    mask, contours, _ = cv2.findContours(input_image, 1, 2)
    moments = cv2.moments(contours[0])
    centroid = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    return centroid


def __get_data_for_mask(detection: dict, score_threshold=0.3, category_dict=None):
    data = []
    for mask, score, det_class, box in zip(detection['detection_masks'], detection['detection_scores'],
                                           detection['detection_classes'], detection['detection_boxes']):
        if score > score_threshold:
            if category_dict is None:
                data.append((mask, score, det_class, box))
            else:
                data.append((mask, score, category_dict[det_class], box))
    return data


def mask_centroids(images: list, **kwargs):
    """
    Get the centroids for each mask in the images

    Arguments:
        :param detections: list of image_process detections
        :type detections: list or np.ndarray
    Keywords:
        :param score_threshold: minimum score to perform operation.
        :type score_threshold: int
        :param category_dict: dictionary containing the labels for the dataset
        :type category_dict: dict
    Returns:
        :return: dictionary containing data and centroid of each detection in each image
    """
    category_dict = None
    score_threshold = 0.3
    images_data = []
    centroids = []
    if 'category_dict' in kwargs.keys():
        category_dict = kwargs['category_dict']
    if 'score_threshold' in kwargs.keys():
        score_threshold = kwargs['score_threshold']
    for detection in images:
        # images_data is a list with a tuple containing the detected data for each image
        images_data.append(__get_data_for_mask(detection, score_threshold=score_threshold, category_dict=category_dict))
    # iterate over the images
    for image in images_data:
        image_centr = []
        # iterate over the detections in each image
        for detection in image:
            centroid = _find_single_centroid(detection[0])
            image_centr.append((detection[0], centroid, detection[1], detection[2], detection[3]))
        centroids.append(image_centr)
    return centroids


def draw(images: list, detections: list, **kwargs):
    """
    Draw the masks.
    Arguments:
        :param images: original images that were used to do the training.
        :type images: list or np.ndarray
        :param detections: objects or beings detected in images.
        :type detections: dict
    Keywords:
        :param draw_centroid: flag that activates drawing of centroid.
        :type draw_centroid: bool
        :param label_and_score: flag that writes label and score in the image title.
        :type label_and_score: bool
        :param bounding_box: flag that activates drawing of the bounding box.
        :type bounding_box: bool
    """
    assert len(images) <= len(detections), 'Size of images must be lesser or equal size of detections. Make sure the' \
                                           'detections are equivalent to the images provided.'
    assert type(images) is list or type(images) is np.ndarray, 'Images must be a list of images or a single image.'
    draw_centroid = None
    label_and_score = None
    bounding_box = None
    if 'draw_centroid' in kwargs.keys():
        assert type(kwargs['draw_centroid']) is bool, 'draw_centroid must be a boolean.'
        draw_centroid = kwargs['draw_centroid']
    if 'label_and_score' in kwargs.keys():
        assert type(kwargs['label_and_score']) is bool, 'label_and_score must be a boolean.'
        label_and_score = kwargs['label_and_score']
    if 'bounding_box' in kwargs.keys():
        assert type(kwargs['bounding_box']) is bool, 'bounding_box must be a boolean.'
        bounding_box = kwargs['bounding_box']

    drawing = []
    score = None
    label = None
    for image, img in zip(images, detections):
        detection_drawing = []
        for detection in img:
            temp_image = image.copy()
            if draw_centroid:
                cv2.circle(temp_image, detection[1], 3, (255, 0, 0), -1)
            if label_and_score:
                score = detection[2]
                label = detection[3]
            if bounding_box:
                upper_left_point = (int(detection[4][1] * image.shape[1]),
                                    int(detection[4][0] * image.shape[0]))
                bottom_right_point = (int(upper_left_point[0] + detection[4][2] * image.shape[0]),
                                      int(upper_left_point[1] + detection[4][3] * image.shape[1]))
                cv2.rectangle(temp_image, upper_left_point, bottom_right_point, (255, 0, 0), 3)
            output = temp_image
            output = cv2.bitwise_and(output, output, mask=detection[0])
            detection_drawing.append((output, score, label))
        drawing.append(detection_drawing)
    fig = 1
    for image in drawing:
        if len(image) < 8:
            figure = plt.figure(fig, figsize=(8, 8))
        else:
            figure = plt.figure(fig, figsize=(14, 14))
        for detection, index in zip(image, range(len(image))):
            plt.figure(fig).subplots_adjust(hspace=.8)
            figure.add_subplot(math.ceil(len(image) / 2 + 1), 2, index + 1)
            plt.title(str(detection[2]) + ': ' + str(detection[1] * 100) + '%')
            plt.imshow(detection[0])
        plt.show()
        fig = fig + 1
