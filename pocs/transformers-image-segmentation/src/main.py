import cv2
import numpy as np
import gradio as gr
from PIL import Image

def no_background(image):
    # Convert the image to the RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize the mask with the same shape as the image
    mask = np.zeros(image_rgb.shape[:2], np.uint8)

    # Initialize the background and foreground models
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # Define the region of interest for the GrabCut algorithm
    # Adjust the rectangle to better capture the cat
    rect = (30,30,image_rgb.shape[1]-60,image_rgb.shape[0]-60)

    # Apply the GrabCut algorithm
    # Increase the number of iterations to allow the algorithm to refine the segmentation
    cv2.grabCut(image_rgb, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    # Create a mask where the sure and likely backgrounds are set to 0 and the sure and likely foregrounds are set to 1
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    # Apply the mask to the image
    image_rgb = image_rgb*mask2[:,:,np.newaxis]

    # Create a mask where the background is set to 1
    background_mask = np.where((mask2==0),1,0).astype('uint8')

    # Apply the background mask to each color channel of the image
    for c in range(3):
        # If the background mask is 1, set the color to green. Otherwise, keep the original color.
        image_rgb[:,:,c] = np.where(background_mask, 0 if c != 1 else 255, image_rgb[:,:,c])

    return image_rgb

gr.Interface(fn=no_background, inputs="image", outputs="image").launch()