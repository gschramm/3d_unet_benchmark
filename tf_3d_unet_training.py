import argparse
import numpy as np
from scipy.ndimage import gaussian_filter

import tensorflow as tf
from tensorflow import keras
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------

def unet3d(input_shape = (128,128,128,1), nfeat = 4, kernel_size = (3,3,3), ds = 2):
    """ simple 3D unet for segmentation

    Parameters
    ----------

    input_shape : tuple
      shape of the input tensor (nx, ny, nz, nchannels)
      nx, ny, nz have to be divisible by 8
      default - (128,128,128,1)

    nfeat: int
      number of features in higher level. gets doubled in every lower level
      default - 4

    kernel_size : tuple
      size of the convolution kernels for Conv3D layers
      default - (3,3,3)

    ds : int
      downsample factor applied in maxpooling layers
      default - 2
    """

    input_layer = keras.layers.Input(input_shape)

    # down sample path 
    conv1 = keras.layers.Conv3D(nfeat * 1, kernel_size, padding="same")(input_layer)
    conv1 = keras.layers.PReLU(shared_axes = [1,2,3])(conv1)
    conv1 = keras.layers.Conv3D(nfeat * 1, kernel_size, padding="same")(conv1)
    conv1 = keras.layers.PReLU(shared_axes = [1,2,3])(conv1)
    pool1 = keras.layers.MaxPooling3D((ds, ds, ds))(conv1)

    conv2 = keras.layers.Conv3D(nfeat * 2, kernel_size, padding="same")(pool1)
    conv2 = keras.layers.PReLU(shared_axes = [1,2,3])(conv2)
    conv2 = keras.layers.Conv3D(nfeat * 2, kernel_size, padding="same")(conv2)
    conv2 = keras.layers.PReLU(shared_axes = [1,2,3])(conv2)
    pool2 = keras.layers.MaxPooling3D((ds, ds, ds))(conv2)

    conv3 = keras.layers.Conv3D(nfeat * 4, kernel_size, padding="same")(pool2)
    conv3 = keras.layers.PReLU(shared_axes = [1,2,3])(conv3)
    conv3 = keras.layers.Conv3D(nfeat * 4, kernel_size, padding="same")(conv3)
    conv3 = keras.layers.PReLU(shared_axes = [1,2,3])(conv3)
    pool3 = keras.layers.MaxPooling3D((ds, ds, ds))(conv3)
    
    # Middle
    convm = keras.layers.Conv3D(nfeat * 8, kernel_size, padding="same")(pool3)
    convm = keras.layers.PReLU(shared_axes = [1,2,3])(convm)
    convm = keras.layers.Conv3D(nfeat * 8, kernel_size, padding="same")(convm)
    convm = keras.layers.PReLU(shared_axes = [1,2,3])(convm)
    convm = keras.layers.Dropout(0.2)(convm)
   
    
    deconv3 = keras.layers.Conv3DTranspose(nfeat * 4, kernel_size, strides=(ds, ds, ds), padding="same")(convm)
    uconv3 = keras.layers.concatenate([deconv3, conv3])
    uconv3 = keras.layers.Conv3D(nfeat * 4, kernel_size, padding="same")(uconv3)
    uconv3 = keras.layers.PReLU(shared_axes = [1,2,3])(uconv3)
    uconv3 = keras.layers.Conv3D(nfeat * 4, kernel_size, padding="same")(uconv3)
    uconv3 = keras.layers.PReLU(shared_axes = [1,2,3])(uconv3)

    deconv2 = keras.layers.Conv3DTranspose(nfeat * 2,  kernel_size, strides=(ds, ds, ds), padding="same")(uconv3)
    uconv2 = keras.layers.concatenate([deconv2, conv2])
    uconv2 = keras.layers.Conv3D(nfeat * 2, kernel_size, padding="same")(uconv2)
    uconv2 = keras.layers.PReLU(shared_axes = [1,2,3])(uconv2)
    uconv2 = keras.layers.Conv3D(nfeat * 2, kernel_size, padding="same")(uconv2)
    uconv2 = keras.layers.PReLU(shared_axes = [1,2,3])(uconv2)

    deconv1 = keras.layers.Conv3DTranspose(nfeat * 1, kernel_size, strides=(ds, ds, ds), padding="same")(uconv2)
    uconv1 = keras.layers.concatenate([deconv1, conv1])
    uconv1 = keras.layers.Conv3D(nfeat * 1, kernel_size, padding="same")(uconv1)
    uconv1 = keras.layers.PReLU(shared_axes = [1,2,3])(uconv1)
    uconv1 = keras.layers.Conv3D(nfeat * 1, kernel_size, padding="same")(uconv1)
    uconv1 = keras.layers.PReLU(shared_axes = [1,2,3])(uconv1)
    
    output_layer = keras.layers.Conv3D(1, (1,1,1), padding="same", activation="sigmoid")(uconv1)
    
    model = keras.Model(input_layer, output_layer)

    return model

#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Dummy 3D unet training')
parser.add_argument('--n_feat', default = 4, type = int, help = 'number of features in highest level of unet')
parser.add_argument('--n', default = 96, type = int, help = 'spatial size of input 3D tensor')
parser.add_argument('--n_train', default = 80, type = int, help = 'number of training data sets')
parser.add_argument('--n_val', default = 10, type = int, help = 'number of training data sets')
parser.add_argument('--batch_size', default = 10, type = int, help = 'batch size used in training')
parser.add_argument('--epochs', default = 30, type = int, help = 'number of epochs in training')

args = parser.parse_args()

n           = args.n
n_feat      = args.n_feat

n_train     = args.n_train
n_val       = args.n_val

batch_size  = args.batch_size
epochs      = args.epochs

if not (n % 8 == 0):
  raise ValueError('spatial size must be divisible by 8')

#-------------------------------------------------------------------------------------------------------

x_train = np.zeros((n_train,n,n,n,1), dtype = np.float32)
y_train = np.zeros((n_train,n,n,n,1), dtype = np.float32)

x_val = np.zeros((n_val,n,n,n,1), dtype = np.float32)
y_val = np.zeros((n_val,n,n,n,1), dtype = np.float32)

# setup random training and validation data
for i in range(n_train):
  x_tmp = gaussian_filter(np.random.randn(n,n,n), 5)
  y_tmp = (x_tmp > 0).astype(np.float32)
  
  # augment contrast
  x_tmp *= (0.5 + 0.5*np.random.rand())
  x_tmp += 0.01*np.random.randn()

  x_train[i,:,:,:,0] = x_tmp
  y_train[i,:,:,:,0] = y_tmp

for i in range(n_val):
  x_tmp = gaussian_filter(np.random.randn(n,n,n), 5)
  y_tmp = (x_tmp > 0).astype(np.float32)
  
  # augment contrast
  x_tmp *= (0.5 + 0.5*np.random.rand())
  x_tmp += 0.01*np.random.randn()

  x_val[i,:,:,:,0] = x_tmp
  y_val[i,:,:,:,0] = y_tmp

x_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train)
x_val = tf.convert_to_tensor(x_val)
y_val = tf.convert_to_tensor(y_val)

#-------------------------------------------------------------------------------------------------------

# setup the model
model = unet3d(input_shape = (n,n,n,1), nfeat = n_feat)

model.compile(optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
              loss      = keras.losses.BinaryCrossentropy())

# train the model
history = model.fit(x_train, y_train,
                    epochs                = epochs,
                    batch_size            = batch_size,
                    validation_data       = (x_val, y_val),
                    validation_batch_size = batch_size,
                    shuffle               = True)

# predict on validation data
pred = model.predict(x_val)
