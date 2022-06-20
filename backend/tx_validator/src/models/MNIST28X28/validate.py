#In[1]
import sys
import random
import h5py
import matplotlib.pyplot as plt
import keras_utils as conv_utils
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Union, Callable, Iterable
from typeguard import typechecked
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D,ZeroPadding3D,Conv3D,BatchNormalization,ReLU,MaxPool3D,Convolution3DTranspose,ELU, Reshape

def neg_Pearson_Loss(predictions, targets):
    '''
    :param predictions: inference value of trained model
    :param targets: target label of input data
    :return: negative pearson loss
    '''
    print(predictions,targets)
    rst = 0
    # Pearson correlation can be performed on the premise of normalization of input data
    predictions = tf.cast(predictions, tf.float32)
    predictions = (predictions - tf.math.reduce_mean(predictions)) / tf.math.reduce_std(predictions)
    targets = (targets - tf.math.reduce_mean(targets)) / tf.math.reduce_std(targets)

    for i in range(1):
        sum_x = tf.math.reduce_sum(predictions[i])  # x
        sum_y = tf.math.reduce_sum(targets[i])  # y
        sum_xy = tf.math.reduce_sum(predictions[i] * targets[i])  # xy
        sum_x2 = tf.math.reduce_sum(tf.math.pow(predictions[i], 2))  # x^2
        sum_y2 = tf.math.reduce_sum(tf.math.pow(targets[i], 2))  # y^2
        N = predictions.shape[1]
        pearson = (N * sum_xy - sum_x * sum_y) / (
            tf.math.sqrt((N * sum_x2 - tf.math.pow(sum_x, 2)) * (N * sum_y2 - tf.math.pow(sum_y, 2))))

        rst += 1 - pearson

    rst = rst / 1
    return rst
# %%
class AdaptivePooling3D(tf.keras.layers.Layer):
    """Parent class for 3D pooling layers with adaptive kernel size.
    This class only exists for code reuse. It will never be an exposed API.
    Args:
      reduce_function: The reduction method to apply, e.g. `tf.reduce_max`.
      output_size: An integer or tuple/list of 3 integers specifying (pooled_dim1, pooled_dim2, pooled_dim3).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    """

    @typechecked
    def __init__(
        self,
        reduce_function: Callable,
        output_size: Union[int, Iterable[int]],
        data_format=None,
        **kwargs,
    ):
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.reduce_function = reduce_function
        self.output_size = conv_utils.normalize_tuple(output_size, 3, "output_size")
        super().__init__(**kwargs)

    def call(self, inputs, *args):
        h_bins = self.output_size[0]
        w_bins = self.output_size[1]
        d_bins = self.output_size[2]
        if self.data_format == "channels_last":
            split_cols = tf.split(inputs, h_bins, axis=1)
            split_cols = tf.stack(split_cols, axis=1)
            split_rows = tf.split(split_cols, w_bins, axis=3)
            split_rows = tf.stack(split_rows, axis=3)
            split_depth = tf.split(split_rows, d_bins, axis=5)
            split_depth = tf.stack(split_depth, axis=5)
            out_vect = self.reduce_function(split_depth, axis=[2, 4, 6])
        else:
            split_cols = tf.split(inputs, h_bins, axis=2)
            split_cols = tf.stack(split_cols, axis=2)
            split_rows = tf.split(split_cols, w_bins, axis=4)
            split_rows = tf.stack(split_rows, axis=4)
            split_depth = tf.split(split_rows, d_bins, axis=6)
            split_depth = tf.stack(split_depth, axis=6)
            out_vect = self.reduce_function(split_depth, axis=[3, 5, 7])
        return out_vect

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    self.output_size[0],
                    self.output_size[1],
                    self.output_size[2],
                    input_shape[4],
                ]
            )
        else:
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    input_shape[1],
                    self.output_size[0],
                    self.output_size[1],
                    self.output_size[2],
                ]
            )

        return shape

    def get_config(self):
        config = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}
# %%
################################
# Reading dataframe
#################################
def read_input(index):
    if len(sys.argv) < (index+1):
        raise Exception('No dataset path found')
# /Users/daeyeolkim/eunsu/work/FLoBC/data/PhysNet_UBFC_test.hdf5
    print("reading dataset...")
    df = h5py.File("/Users/daeyeolkim/eunsu/work/FLoBC/data/PhysNet_UBFC_train_1.hdf5", "r")
    print("dataset reading success")
    # df = pd.read_csv(sys.argv[index])
    # df = pd.read_csv("data.csv")
    if len(df) == 0:
        raise Exception('Empty dataset')
    return df

# %%
################################
# Reshaping input
################################
# %%
def reshapeData(index):
    df = read_input(index)

    idx = random.randint(0,10)
    cnt = 0

    video_data = []
    label_data = []

    for key in df.keys():
        if cnt == idx:
            video_data.extend(df[key]['preprocessed_video'])
            label_data.extend(df[key]['preprocessed_label'])
            break
        else:
            cnt+=1
    df.close()
    # df = df.sample(int(0.3*len(df)))
    # label = df.iloc[:, 0]
    # label = label.to_numpy()
    # df = df.drop(df.columns[0], axis=1)
    # df = df.values.reshape(df.shape[0], *INPUT_SHAPE)
    # Making sure that the values are float so that we can get decimal points after division
    # df = df.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    # df /= 255
    return video_data, label_data
# %%
def createModel():
    # Creating a Sequential Model and adding the layers
    model = Sequential()

    model.add(ZeroPadding3D(padding=(0,2,2),input_shape=(32,128,128,3)))
    model.add(Conv3D(filters=16,kernel_size=(1,5,5),strides=(1,1,1)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(MaxPool3D(pool_size=(1,2,2), strides=(1,2,2)))

    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(ZeroPadding3D(padding=(1, 1, 1)))
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Convolution3DTranspose(filters=64,kernel_size=(4,1,1),strides=(2,1,1),padding='same'))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Convolution3DTranspose(filters=64,kernel_size=(4,1,1),strides=(2,1,1),padding='same'))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(AdaptivePooling3D(tf.reduce_max,(32,1,1)))

    model.add(Conv3D(1,kernel_size=(1,64,64),strides=(1,1,1),padding='same')),
    model.add(Reshape((-1,)))

    model.compile(optimizer='adam', loss=neg_Pearson_Loss, metrics=['neg_Pcc'])
    return model

# %%
def evaluateModel(model, data_test, label_test):
  results = model.evaluate(data_test, label_test, verbose=2)
  # Return accuracy
  return results[1]


# %%
def rebuildModel(flat_model):
  new_model = createModel()
  start = 0
  for i in range (0, len(new_model.layers)):
      bound = np.array(new_model.layers[i].get_weights()).size
      weights = []
      for j in range (0, bound):
        size = (new_model.layers[i].get_weights()[j]).size
        arr = np.array(flat_model[start:start+size])
        arr = arr.reshape(new_model.layers[i].get_weights()[j].shape)
        weights.append(arr)
        start += size
      if (bound > 0):
        new_model.layers[i].set_weights(weights)
  return new_model
# %%

################################
# Validation score
################################
def compute_validation_score(flat_model, data_dir):
  data_test, label_test = reshapeData(data_dir)
  model = rebuildModel(flat_model)
  result = evaluateModel(model, data_test, label_test)
  print(result)
  return result
# %%