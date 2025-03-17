import tensorflow as tf
from tensorflow.keras import layers, Model

import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


print("Versión de TensorFlow:", tf.__version__)
print("GPU disponible:", tf.config.list_physical_devices('GPU'))

# gpus = tf.config.experimental.list_physical_devices('GPU')

# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
        

class ModelManager:
    def __init__(self):
        pass

    def unet(self, input_shape):
        # Entrada
        inputs = layers.Input(shape=input_shape)  # Formato: (H, W, 2)

        # Encoder
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        # Cuello de botella
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)

        # Decoder
        up4 = layers.UpSampling2D(size=(2, 2))(conv3)
        up4 = layers.Conv2D(128, 2, activation='relu', padding='same')(up4)
        merge4 = layers.concatenate([conv2, up4], axis=-1)
        conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge4)
        conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)

        up5 = layers.UpSampling2D(size=(2, 2))(conv4)
        up5 = layers.Conv2D(64, 2, activation='relu', padding='same')(up5)
        merge5 = layers.concatenate([conv1, up5], axis=-1)
        conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge5)
        conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)

        # Salida
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)  # Salida en el rango [0, 1]

        # Crear el modelo
        model = Model(inputs=inputs, outputs=outputs)
        return model



