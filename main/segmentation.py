import cv2
from cellpose import models
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    output_dir = 'output/'
    os.makedirs(output_dir, exist_ok=True)

    image_path = 'images/1.JPG'

    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load the image from {image_path}")
    else:
        print("Image loaded successfully.")

    model = models.Cellpose(gpu=False)

    channels = [0, 0]

    if image is not None:
        print("Image shape:", image.shape)

    
    # Apply averaging filter to smooth the image
    average_blur = cv2.blur(image, (5, 5))

    output_path = os.path.join(output_dir, 'average_blur.jpg')
    cv2.imwrite(output_path, average_blur)
    print(f"average_blur saved at: {output_path}")
    print("average_blur shape:", average_blur.shape)

    gray = cv2.cvtColor(average_blur, cv2.COLOR_BGR2GRAY)

    output_path = os.path.join(output_dir, 'gray.jpg')
    cv2.imwrite(output_path, gray)
    print(f"gray saved at: {output_path}")
    print("gray shape:", gray.shape)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    output_path = os.path.join(output_dir, 'equalized.jpg')
    cv2.imwrite(output_path, equalized)
    print(f"equalized saved at: {output_path}")
    print("equalized shape:", equalized.shape)

    # Apply morphological filters
    kernel = np.ones((5, 5), np.uint8)
    morphology = cv2.morphologyEx(equalized, cv2.MORPH_CLOSE, kernel)
    morphology = cv2.morphologyEx(morphology, cv2.MORPH_OPEN, kernel)

    output_path = os.path.join(output_dir, 'morphology.jpg')
    cv2.imwrite(output_path, morphology)
    print(f"morphology saved at: {output_path}")

    print("morphology shape:", morphology.shape)

    # Perform segmentation
    masks, flows, styles, diams = model.eval(morphology, channels=channels)

    # masks = np.zeros(image.shape, dtype=np.uint8)
    # masks = cv2.circle(masks, (260, 300), 225, (255,255,255), -1) 

    if masks is None:
        print("Error: Segmentation failed.")


    # # Save the masks
    # for i in range(len(masks)):
    #     mask_i = masks[i]
    #     #cv2.imshow(f'Mask {i + 1}', mask_i)

    output_path = os.path.join(output_dir, 'masks.jpg')
    cv2.imwrite(output_path, masks)
    print(f"masks image saved at: {output_path}")
    print("mask shape:", masks.shape)
    # Apply the masks to the original image
    segmented_image = np.zeros_like(image) 
    print("segmented_image shape:", segmented_image.shape)

    rows,cols,_ = segmented_image.shape

    # segmented_image = cv2.bitwise_and(image,image,mask = masks)
    # mask_inv = masks  
    segmented_image = image* [0, 255, 0] * masks

    # for mask in masks:
    #     # mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize the mask to match the image size
    #     # mask_expanded = np.expand_dims(mask_resized, axis=-1)
    #     segmented_image[1 > 0] = [0, 255, 0]  # Set masked pixels to green (adjust as needed)

    # Save the segmented image
    output_path = os.path.join(output_dir, 'segmented_image.jpg')
    cv2.imwrite(output_path, segmented_image)
    print(f"Segmented image saved at: {output_path}")
except Exception as e:
    print(f"An error occurred: {e}")