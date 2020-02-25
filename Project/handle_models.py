import cv2
import numpy as np


def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''
    # Extract only the second blob output (keypoint heatmaps)
    second_blob = output['Mconv7_stage2_L2']
    
    # Resize the heatmap back to the size of the input
    heatmap = np.zeros([second_blob.shape[1], input_shape[0], input_shape[1]])
    for h in range(len(second_blob[0])):
        heatmap[h] = cv2.resize(second_blob[0][h], input_shape[0:2][::-1])
    print(heatmap.shape)
    
    return heatmap


def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "POSE":
        return handle_pose
    else:
        return None


'''
The below function is carried over from the previous exercise.
You just need to call it appropriately in `app.py` to preprocess
the input image.
'''
def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image