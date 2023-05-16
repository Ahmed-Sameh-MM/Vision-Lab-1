import cv2
import numpy as np


def feature_extraction_sift(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def feature_matching(image1, image2):
    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = feature_extraction_sift(image1)
    keypoints2, descriptors2 = feature_extraction_sift(image2)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image


def show_feature_matching(image_1_path, image_2_path):
    # Example usage
    image_1 = cv2.imread(image_1_path)
    image_2 = cv2.imread(image_2_path)

    cv2.imshow('Original Image 1', image_1)
    cv2.imshow('Original Image 2', image_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Feature extraction using SIFT
    key_points_1, descriptors_1 = feature_extraction_sift(image_1)
    key_points_2, descriptors_2 = feature_extraction_sift(image_2)

    # Feature matching
    matched_image = feature_matching(image_1, image_2)
    cv2.imshow('Feature Matching', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_scaling(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    new_dimensions = (width, height)
    scaled_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)
    return scaled_image


def image_rotation(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h))
    return rotated_image


def image_translation(image, x, y):
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    translated_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return translated_image


def show_all_transformations(image_path):

    image = cv2.imread(image_path)

    # Image scaling
    scaled_image = image_scaling(image, 50)
    cv2.imshow('Scaled Image', scaled_image)

    # Image rotation
    rotated_image = image_rotation(image, 45)
    cv2.imshow('Rotated Image', rotated_image)

    # Image translation
    translated_image = image_translation(image, 50, 50)
    cv2.imshow('Translated Image', translated_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
show_feature_matching('Images/image_1.jpg', 'Images/helmy_face_1.jpg')

show_all_transformations('Images/image_1.jpg')

# Example usage
show_feature_matching('Images/image_2.jpg', 'Images/helmy_face_2.jpg')

show_all_transformations('Images/image_2.jpg')

# Example usage
show_feature_matching('Images/image_3.jpg', 'Images/cat_face.jpg')

show_all_transformations('Images/image_3.jpg')

if __name__ == '__main__':
    print('Lab Init')
