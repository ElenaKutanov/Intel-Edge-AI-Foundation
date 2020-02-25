import argparse
import cv2
from inference import Network
import numpy as np
import sys

INPUT_STREAM = ""
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='BLUE')
    optional.add_argument("-ct", help=ct_desc, default=0.5)
    args = parser.parse_args()

    return args


def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']

def get_mask(processed_output):
    '''
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    '''
    # Create an empty array for other color channels of mask
    empty = np.zeros(processed_output.shape)
    # Stack to make a Green mask where text detected
    mask = np.dstack((empty, processed_output, empty))

    return mask


def handle_pose(output, input_shape):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''
    # Extract only the second blob output (keypoint heatmaps)
    heatmaps = output['Mconv7_stage2_L2']
    # Resize the heatmap back to the size of the input
    # Create an empty array to handle the output map
    out_heatmap = np.zeros([heatmaps.shape[1], input_shape[0], input_shape[1]])
    # Iterate through and re-size each heatmap
    for h in range(len(heatmaps[0])):
        out_heatmap[h] = cv2.resize(heatmaps[0][h], input_shape[0:2][::-1])

    return out_heatmap

def create_output_image(image, output):
    # Remove final part of output not used for heatmaps
    # Get only pose detections above 0.5 confidence, set to 255
    unicorn = cv2.imread('./images/unicorn.png')
    unicorn = unicorn[:, :, :3]
    output = output[:-1]
    for c in range(len(output)):
        output[c] = np.where(output[c] > 0.5, 255, 0)
    # Sum along the "class" axis
    output = np.sum(output, axis=0)
    # Get semantic mask
    pose_mask = get_mask(output)
    # Combine with original image
    image = cv2.addWeighted(image, 0.5, pose_mask.astype(np.uint8), 0.5, 0)
    xmin, xmax, ymin, ymax = calc_hw(pose_mask)
    unicorn = cv2.resize(unicorn, dsize=(xmax-xmin, ymax-ymin), interpolation=cv2.INTER_NEAREST)
    unicorn = np.where((unicorn != 0), unicorn, image[ymin:ymax, xmin:xmax])
    image[ymin:ymax, xmin:xmax] = unicorn
    return image

def calc_hw(mask):
    cm = mask[:, :, 1]
    if not np.any(cm == False):
        return 1, 3, 2, 3
    m = cm == 255
    columns_indices = np.where(np.any(m, axis=0))[0]
    rows_indices = np.where(np.any(m, axis=1))[0]
    if len(columns_indices) == 0:
        return 1, 3, 2, 3
    first_column_index, last_column_index = columns_indices[0], columns_indices[-1]
    first_row_index, last_row_index = rows_indices[0], rows_indices[-1]
    #print(first_column_index, last_column_index, first_row_index, last_row_index)

    return first_column_index, last_column_index, first_row_index, last_row_index

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

def infer_on_video(args):
    # Convert the args for color and confidence
    args.c = convert_color(args.c)
    args.ct = float(args.ct)

    ### Initialize the Inference Engine
    plugin = Network()

    ### Load the network model into the IE
    n, c, h, w = plugin.load_model(args.m, args.d)

    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(0)

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        cv2.imshow('Input', frame)

        ### Pre-process the frame
        p_frame = preprocessing(frame, h, w)

        ### Perform inference on the frame
        plugin.async_inference(p_frame)

        ### Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            processed_output = handle_pose(result, frame.shape)
            frame2 = create_output_image(frame, processed_output)
            ### Update the frame to include detected bounding boxes
            cv2.imshow('Output', frame2)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
