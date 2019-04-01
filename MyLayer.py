# 自定义层

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.core import activations

class WeightedAdd(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(WeightedAdd, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[2], self.output_dim), initializer='uniform', trainable=True)
        super(WeightedAdd, self).build(input_shape)
    
    def call(self, x):
        return K.dot(x, self.kernel)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

# tanh(hwt + b)
class AttentionScore(Layer):
    def __init__(self, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        self.activation = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        super(AttentionScore, self).__init__(**kwargs)
    
    def build(self, input_shape):
        print('AttentionScore input shape:\n', input_shape)
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[0][2], input_shape[1][1]), initializer=self.kernel_initializer, trainable=True)
        self.bias = self.add_weight(name='bias', shape=(input_shape[0][1], ), initializer=self.bias_initializer, trainable=True)
        super(AttentionScore, self).build(input_shape)
    
    def call(self, x):
        output0 = K.dot(x[0], self.kernel)
        output1 = K.batch_dot(output0, x[1])
        output2 = K.bias_add(output1, self.bias)
        output3 = self.activation(output2)
        return output3
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],input_shape[0][1])