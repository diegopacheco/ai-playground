import cv2

def downscale_image(image_path, output_path, scale_percent):
    # Load the original image
    original_image = cv2.imread(image_path)

    # Calculate the new dimensions
    width = int(original_image.shape[1] * scale_percent / 100)
    height = int(original_image.shape[0] * scale_percent / 100)
    new_dimensions = (width, height)
    new_dimensions = (800, 600)

    # Resize the image
    downscaled_image = cv2.resize(original_image, new_dimensions, interpolation = cv2.INTER_CUBIC)

    # Save the downscaled image
    cv2.imwrite(output_path, downscaled_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

# Use the function
downscale_image('image-original.jpg', 'downscaled_image.jpg', 100)
print('Image downscaled successfully!')