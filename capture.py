import cv2
import numpy as np

def geometric_transform(reference_img, captured_img, save_filename):
    """
    Method: Geometrically corrects the warped image captured with camera.

    Arguments:
        - Reference image
        - Captured image
        - File name of the corrected image

    Algorithm:
        - Load reference image and captured image.
        - Convert both images to grayscale.
        - Match features from the captured image, to the reference image and store the coordinates of the corresponding key points.
          Keypoints are simply the selected few points that are used to compute the transform (generally points that stand out),
          and descriptors are histograms of the image gradients to characterize the appearance of a keypoint.
          Uses ORB (Oriented FAST and Rotated BRIEF) library of openCV, which provides both key points as well as their associated descriptors.

        - Match the key points between the two images. Uses BFMatcher, which is a brute force matcher.
          BFMatcher.match() retrieves the best match between two images.

        - Pick the top matches, and remove the noisy matches.
        - Find the homomorphy transform.
        - Apply this transform to the original unaligned image to get the output image.
    """
    img1_color = cv2.imread(captured_img)  # Captured color image
    img2_color = cv2.imread(reference_img)  # Reference color image

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # Create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
    #     print(matches)

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

    # Save the transformed image output.
    cv2.imwrite(save_filename, transformed_img)

## Call the function

# Specify the path for the reference image
#reference_img = "C:/Research/COdes/DeepD2C/21DEC04/input_images/3.jpg"
reference_img = r"C:\Research\COdes\DeepD2C\experiment B\test_images\1_256.jpg"

# Specify the path for the captured image
#captured_img = "C:/Research/COdes/DeepD2C/21DEC04/captured_images/before_geometric/3_captured.jpg"
captured_img = r"C:\Research\COdes\DeepD2C\experiment B\Distance\Captured at 10 cm\img0001.jpg"
# Specify the file name and saving directory
#save_filename = "C:/Research/COdes/DeepD2C/21DEC04/captured_images/after_geometric/3_corrected.png"
save_filename = r"C:\Research\COdes\DeepD2C\experiment B\Distance\corrected\corrected_img001.png"

geometric_transform(reference_img, captured_img, save_filename)