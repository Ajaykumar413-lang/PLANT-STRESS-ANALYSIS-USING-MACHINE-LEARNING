import cv2
import numpy as np

def extract_features(image):
    # Resize image
    img = cv2.resize(image, (256, 256))

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mean Intensity (brightness)
    mean_intensity = np.mean(img)

    # Green Ratio using HSV
    # Lower and upper green values
    lower_green = np.array([36, 40, 40])
    upper_green = np.array([86, 255, 255])

    # Create mask for green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculate proportion of green pixels
    green_ratio = cv2.countNonZero(mask) / (256 * 256)

    # Texture (leaf surface detail)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(gray, cv2.CV_64F).var()

    return [mean_intensity, green_ratio, texture]