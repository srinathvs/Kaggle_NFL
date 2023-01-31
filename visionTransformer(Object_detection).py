# TODO : Run and test it on MSCoCo, create Jupyter notebooks to refernce code written here
# refer to create Vit detector function to understand the complete model.


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import cv2
import os
import scipy.io
import shutil


def mlp(x, hidden_layers, dropout_rate):
    for units in hidden_layers:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)

    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": input_shape,
                "patch_size": patch_size,
                "num_patches": num_patches,
                "projection_dim": projection_dim,
                "num_heads": num_heads,
                "transformer_units": transformer_units,
                "transformer_layers": transformer_layers,
                "mlp_head_units": mlp_head_units,
            }
        )
        return config

    def call(self, images):
        batch_size = tf.shape(images[0])
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID")

        return tf.reshape(patches, [batch_size, -1, patches.shape[-1]])


class Patch_Encoder(layers.Layer):
    def __init__(self, num_patches, projection_dims):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dims)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dims)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": input_shape,
                "patch_size": patch_size,
                "num_patches": num_patches,
                "projection_dim": projection_dim,
                "num_heads": num_heads,
                "transformer_units": transformer_units,
                "transformer_layers": tranformer_layers,
                "mlp_head_units": mlp_head_units
            }
        )

        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_detector(
        input_shape,
        patch_size,
        num_patches,
        projection_dim,
        num_heads,
        transformer_units,
        transformer_layers,
        mlp_head_units
):
    inputs = layers.Input(shape=input_shape)

    patches = Patches(patch_size)(inputs)

    encoded_patches = Patch_Encoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = layers.MultiHeadAttention(num_heads, projection_dim, dropout=.1)(x1, x1)

        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        x3 = mlp(x3, transformer_units, dropout_rate=.1)

        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(.3)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=.3)

    bounding_box = layers.Dense(4)(features)

    return keras.Model(inputs=inputs, outputs=bounding_box)


def run_experiment(model, learning_rate, weight_decay, batch_size, num_epochs, x_train, y_train):

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    # Compile model.
    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())

    checkpoint_filepath = "logs/"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[
            checkpoint_callback,
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ],
    )

    return history

if __name__ == '__main__':

     # Trying out some sample data to initialize ( cannot create patches with no data ).
    image_size = 256
    patch_size = 32
    input_shape = (image_size, image_size, 3)  # input image shape
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 32
    num_epochs = 100
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    # Size of the transformer layers
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    transformer_layers = 4
    mlp_head_units = [2048, 1024, 512, 64, 32]  # Size of the dense layers

    history = []
    num_patches = (image_size // patch_size) ** 2

    vit_object_detector = create_vit_detector(
        input_shape,
        patch_size,
        num_patches,
        projection_dim,
        num_heads,
        transformer_units,
        transformer_layers,
        mlp_head_units,
    )
