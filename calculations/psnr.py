from math import log10, sqrt
import cv2
import numpy as np


def calculate_psnr(original, compressed):
    original = cv2.imread(original)
    compressed = cv2.imread(compressed, 1)
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def main():
    original = cv2.imread("C:/DL lecture slides/DeepD2C/images_test/4.jpg")
    compressed = cv2.imread("C:/DL lecture slides/DeepD2C/experiments/Jul-25-12-39-53-PM/encoded-1658720393/4_encoded_.png", 1)
    value = psnr(original, compressed)
    print(f"PSNR value is {value} dB")


if __name__ == "__main__":
    main()