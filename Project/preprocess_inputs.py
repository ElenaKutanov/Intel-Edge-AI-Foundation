import cv2
import numpy as np

def prep(img, h, w):
    #print('1:', img.shape)
    img = cv2.resize(img, (h, w))
    #print('2:', img.shape)
    img = img.transpose(2, 0, 1)
    #print('3:', img.shape)
    img = img.reshape(1, 3, h, w)
    #print('4:', img.shape)

    return img


def pose_estimation(input_image):
    '''
    Given some input image, preprocess the image so that
    it can be used with the related pose estimation model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    '''
    preprocessed_image = np.copy(input_image)

    # Preprocess the image for the pose estimation model
    preprocessed_image = prep(preprocessed_image, 256, 456)

    return preprocessed_image
