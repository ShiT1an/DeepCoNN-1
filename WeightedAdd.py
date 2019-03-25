# 自定义带权相加的输出层

from keras import backend as K
from keras.engine.topology import Layer

class WeightedAdd(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(WeightedAdd, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[2], self.output_dim), initializer='uniform', trainable=True)
        super(WeightedAdd, self).build(input_shape)
    
    def call(self, x):
        print(x.shape, self.kernel.shape)
        return K.dot(x, self.kernel)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)