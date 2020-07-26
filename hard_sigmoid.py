import tensorflow as tf
def hard_sigmoid(x):
    return tf.keras.backend.clip((x+1.)/2.,0,1)