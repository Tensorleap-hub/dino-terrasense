import os
from typing import List
from PIL import Image
import tensorflow as tf
import numpy as np

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse

from dino_terrasense.config import config
from dino_terrasense.utils.general_utils import preprocess_image


# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    train_images_paths = os.listdir(config['train_images_dir'])
    val_images_paths = os.listdir(config['val_images_dir'])
    train = PreprocessResponse(length=len(train_images_paths), data=train_images_paths)
    val = PreprocessResponse(length=len(val_images_paths), data=val_images_paths)
    response = [train, val]
    return response


# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image. 
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    img = Image.open(preprocess.data[idx])
    img = preprocess_image(img)
    return img.astype('float32')


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return np.zeros(1)


def placeholder_loss(gt: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(y_pred, axis=-1) * 0


# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.set_preprocess(function=preprocess_func)
leap_binder.set_input(function=input_encoder, name='image')
leap_binder.set_ground_truth(function=gt_encoder, name='dummy')
leap_binder.add_custom_loss(placeholder_loss, 'zero_loss')

if __name__ == '__main__':
    leap_binder.check()