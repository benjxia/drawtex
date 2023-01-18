import tensorflow as tf
from tensorflow.python import keras
from keras import Model, layers

class DT_Model_tf(Model):
    def __init__(self):
        super(DT_Model_tf, self).__init__()
        
        self.conv1 = layers.Conv2D(filters=64,
                                   kernel_size=4,
                                   activation="relu",
                                   input_shape=(45, 45, 1)
        )

        # self.norm1 = layers.norm
    
    def call(self, x, train=False):
        pass
model: DT_Model_tf = DT_Model_tf()
# model.add
model.build(input_shape=(45, 45, 1))
print(model.summary())