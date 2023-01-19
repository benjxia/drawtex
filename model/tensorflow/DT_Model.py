import tensorflow as tf
from tensorflow.python import keras
from keras import Model, layers

class DT_Model_tf(Model):
    def __init__(self):
        super(DT_Model_tf, self).__init__()
        
        self.conv1 = layers.Conv2D(filters=64,
                                   kernel_size=4,
                                   activation="relu",
                                    )

        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))

        # self.norm1 = layers.norm
        self.conv2 = layers.Conv2D(filters=128,
                                   kernel_size=4,
                                   activation="relu",
                                    )
        

    def call(self, x, train=False):
        x = tf.reshape(x, [-1, 45, 45, 1])
        x = self.conv1(x)
        # x = self.pool1(x)
        return x


model: DT_Model_tf = DT_Model_tf()
# model.add
model.build(input_shape=(45, 45, 1))
print(model.summary())