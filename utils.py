import matplotlib.pyplot as plt
import tensorflow as tf
from imgaug import augmenters as iaa
from  imgaug.augmentables.segmaps import SegmentationMapsOnImage
import os
import numpy as np
import imgaug as ia
import cv2

def show_model_predictions(model, test_images, test_masks, title_suffix, index=0, show_display = False):
    """
    Displays the test set image, ground truth mask, and model prediction.
    
    Args:
    - model: The trained or untrained model for predictions.
    - test_images: The set of test images.
    - test_masks: The ground truth masks for the test images.
    - title_suffix: A string to indicate whether it's before or after training.
    - index: The index of the image to be displayed. Default is 5.
    """
    test_pred = model(test_images)
    test_out = tf.argmax(test_pred[index], axis=-1)

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(test_images[index])
    plt.axis("off")
    plt.title("Test Set Image")

    plt.subplot(1, 3, 2)
    plt.imshow(test_masks[index])
    plt.axis("off")
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(test_out)
    plt.axis("off")
    plt.title("Model Prediction")

    plt.tight_layout()
    plt.savefig(f"images/prediction_{title_suffix.lower().replace(' ', '_')}.png", bbox_inches='tight', pad_inches=.1, dpi=300)
    print(f"Figure Saved: model_prediction_{title_suffix.lower().replace(' ', '_')}.png")
    if show_display == True:
        plt.show()


def load_data(path: str):
    images_path = os.path.join(path, "inputs")
    masks_path = os.path.join(path, "masks")

    assert len(os.listdir(images_path)) == len(os.listdir(masks_path)), "The number of images and masks are not equal."
    number_of_images = (len(os.listdir(images_path)))
    
    images = []
    masks = []

    for image, mask in zip(os.listdir(images_path), os.listdir(masks_path)):

        image = cv2.imread(os.path.join(images_path, image))
        mask = cv2.imread(os.path.join(masks_path, mask), cv2.IMREAD_GRAYSCALE)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        mask = cv2.resize(mask, (128, 128))

        images.append(image)
        masks.append(mask)

    print(f"{number_of_images} images and their masks were loaded.")

    images = np.array(images)
    masks = np.array(masks)
    masks = np.where(masks < 128, 0, 1)
    
    return images, masks
    


def data_augmentation(image, mask, show_display=False):
    ia.seed(2)

    # Ensure mask is single-channel and same size as image
    if len(mask.shape) == 3 and mask.shape[2] > 1:
        mask = mask[:, :, 0]  # Take the first channel if mask is multi-channel
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize mask to match image size

    segmap = SegmentationMapsOnImage(mask, shape=image.shape)

    seq = iaa.Sequential([
        iaa.ElasticTransformation(alpha=(0, 30), sigma=(4, 6)),
        iaa.Affine(scale=(0.5, 2), rotate=(-90, 90)),
        iaa.GammaContrast((0.5, 2))
    ], random_order=True)
    
    image_aug, segmap_aug = seq(images=[image], segmentation_maps=[segmap])
    
    if show_display:
        plt.figure(figsize=(15, 10))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title("Original Image")

        # Original image with mask
        plt.subplot(1, 3, 2)
        plt.imshow(np.array(segmap.draw_on_image(np.array(image, dtype="uint8"))[0], dtype='uint8'))
        plt.axis("off")
        plt.title("Original Image with Mask")

        # Augmented image with augmented mask
        plt.subplot(1, 3, 3)
        plt.imshow(segmap_aug[0].draw_on_image(np.array(image_aug[0], dtype="uint8"))[0])
        plt.axis("off")
        plt.title("Augmented Image with Mask")
        
        plt.savefig("images/augmentation.png", bbox_inches='tight', pad_inches=.1, dpi=300)
        if show_display == True:
            plt.show()
        
    return image_aug[0], segmap_aug[0].get_arr()
