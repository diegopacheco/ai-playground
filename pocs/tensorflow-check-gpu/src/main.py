import tensorflow as tf
tf.config.list_physical_devices("GPU")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())