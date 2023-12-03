# %%

import numpy as np
from tensorflow import keras
from keras import layers
import cv2
import os
import imgaug as ia
from imgaug import augmenters as iaa
from  imgaug.augmentables.segmaps import SegmentationMapsOnImage
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import show_model_predictions, load_data, data_augmentation

# %%
TRAIN_PATH = os.path.join("datasets", "data-science-bowl-2018-2", "train")
TEST_PATH = os.path.join("datasets", "data-science-bowl-2018-2", "test")
SHOW_DISPLAY = False
EPOCHS = 100
BATCH_SIZE = 32
OUTPUT_CHANNELS = 2

# %%

# Load the training and test set
train_images, train_masks = load_data(TRAIN_PATH)
test_images, test_masks = load_data(TEST_PATH)

# Augment the data
_, _ = data_augmentation(train_images[0], train_masks[0], SHOW_DISPLAY)
images_aug, masks_aug = zip(*[data_augmentation(train_image, train_mask) for train_image, train_mask in zip(train_images, train_masks)])

# %%
train_images = np.concatenate([train_images, images_aug], axis=0)
train_masks = np.concatenate([train_masks, masks_aug], axis=0)

plt.figure(figsize=(15, 20))
n_samples = 3

for i in (range(n_samples)):
    plt.subplot(2, n_samples, i+1)
    plt.imshow(train_images[i])
    plt.axis("off")
    plt.title(f"Image {i}")

    plt.subplot(1, n_samples, i+1)
    plt.imshow(train_masks[i])
    plt.axis("off")
    plt.title(f"Mask {i}")

plt.savefig("images/training_data_sample.png", bbox_inches='tight', pad_inches=.1, dpi=300)

normalized_train_images = train_images/255
normalized_test_images = test_images/255

# %%
# We are using pre-trained mobilenetv2's initial layers for feature extraction
mobile_v2_model = keras.applications.MobileNetV2(
    input_shape=[128, 128, 3], include_top=False
)

# List down some activation layers
layer_names = [
    "block_1_expand_relu",  # 64x64
    "block_3_expand_relu",  # 32x32
    "block_6_expand_relu",  # 16x16
    "block_13_expand_relu",  # 8x8
    "block_16_project",  # 4x4
]

encoder_outputs = [mobile_v2_model.get_layer(name).output for name in layer_names]

# Define the feature extraction model
encoder = keras.Model(inputs=mobile_v2_model.input, outputs=encoder_outputs[-1])
encoder.trainable = True

# Create decoder(up_stack) layers
up_layer_1 = layers.Conv2DTranspose(512, 3, strides=2, padding="same", kernel_initializer='he_normal', name='up_1') # 4 x 4 -> 8 x 8
up_layer_2 = layers.Conv2DTranspose(256, 3, strides=2, padding="same", kernel_initializer='he_normal', name= 'up_2') # 8 x 8 -> 16 x 16
up_layer_3 = layers.Conv2DTranspose(128, 3, strides=2, padding="same", kernel_initializer='he_normal', name= 'up_3') # 16 x 16 -> 32 x 32
up_layer_4 = layers.Conv2DTranspose(64, 3, strides=2, padding="same", kernel_initializer='he_normal', name= 'up_4') # 32 x 32 -> 64 x 64
decoder_outputs = ([up_layer_1, up_layer_2, up_layer_3, up_layer_4])

# %%


def U_Net(output_channels):
    inputs = encoder.input

    # Extract encoder outputs
    encoder_outs = encoder(inputs)
    # Start from the deepest layer
    x = encoder_outs

    # Reverse the order of encoder outputs for concatenation
    encoder_reversed = list(reversed(encoder_outputs[:-1]))
    # Apply up-sampling and concatenate
    for encoder_out, decoder_out in zip(encoder_reversed, decoder_outputs):
        x = decoder_out(x)  # Up-sample
        x = layers.Concatenate()([x, encoder_out])  # Concatenate

    # Final convolutional layer
    final_layer = layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same')
    x = final_layer(x)

    unet = keras.Model(inputs=inputs, outputs=x)
    return unet

# %%
# Create the U-Net model
model = U_Net(output_channels=2)

# Defining Loss and Optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.AdamW(learning_rate=0.0005)

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics="accuracy")

# Display the model prediction before training
show_model_predictions(model, normalized_test_images, test_masks, "Before Training", show_display=SHOW_DISPLAY)

# Fit the model
history = model.fit(normalized_train_images, train_masks, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=[normalized_test_images, test_masks])

# Save model summary
with open('images/model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
# Save the model
model.save("saved_models/.model.tf", save_format="tf")

# Display the model prediction after training
show_model_predictions(model, normalized_test_images, test_masks, "after Training", show_display = SHOW_DISPLAY)

# %%
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("images/history.png", bbox_inches='tight', pad_inches=.1, dpi=300)
if SHOW_DISPLAY == True:
    plt.show()
