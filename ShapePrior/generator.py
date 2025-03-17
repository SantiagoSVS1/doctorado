import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence

from skimage.transform import resize

class DataGenerator(Sequence):
    def __init__(self, image_dir, label_dir, batch_size, input_shape, ventricle_id=3):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.ventricle_id = ventricle_id
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.indexes = np.arange(len(self.image_files))

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        X = np.zeros((len(batch_indexes), *self.input_shape))
        y = np.zeros((len(batch_indexes), *self.input_shape))
        
        for i, idx in enumerate(batch_indexes):
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            label_path = os.path.join(self.label_dir, self.label_files[idx])
            
            img = io.imread(img_path, as_gray=True)
            label = io.imread(label_path, as_gray=True)
            label = (label == self.ventricle_id).astype(np.float32)
            
            img = resize(img, self.input_shape[:2])
            label = resize(label, self.input_shape[:2])
            ventricle_mask = label
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            

            
            img = np.expand_dims(img, axis=-1)
            ventricle_mask = np.expand_dims(ventricle_mask, axis=-1)
            
            X[i] = img
            y[i] = ventricle_mask
        
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def test_plot(self, index):
        '''function that takes the first element of the batch and plot image and label in the same figure using __getitem__'''
        X, y = self.__getitem__(index)
        img = X[0, ..., 0]
        label = y[0, ..., 0]
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(label, cmap='gray')
        plt.title('Label')
        plt.show()
