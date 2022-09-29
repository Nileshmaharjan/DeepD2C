import cv2
import numpy as np
import re

images_to_be_aligned_path = 'C:/DL lecture slides/DeepD2C/ber_test_screen_brightness/resolution/flash/1280x720/not_aligned'
referenced_images_path = 'C:/DL lecture slides/DeepD2C/ber_test_screen_brightness/reference_images'
aligned_images_path = 'C:/DL lecture slides/DeepD2C/ber_test_screen_brightness/resolution/flash/1280x720/aligned'

import os
from pathlib import Path


images_to_be_aligned_path_modified = os.listdir(images_to_be_aligned_path)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

images_to_be_aligned_path_modified.sort(key=natural_keys)
referenced_images_path_modified = sorted(Path(images_to_be_aligned_path).iterdir(), key=os.path.getmtime)


for index, file in enumerate(images_to_be_aligned_path_modified):
    unaligned_image = os.path.join(images_to_be_aligned_path, file)
    reference_image = referenced_images_path + "/" + str(index + 1) + '.jpg'
    print(unaligned_image)
    print(reference_image)

    img1_color = cv2.imread(unaligned_image)  # Image to be aligned.
    img2_color = cv2.imread(reference_image)  # Reference image.

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    # (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    # matches.sort(key = lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                                          homography, (width, height))

    # Save the output.
    filename =  str(index + 1) + '.jpg'
    cv2.imwrite(os.path.join(aligned_images_path, filename), transformed_img)
