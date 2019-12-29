import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from typing import Callable


        
class MNISTSequence(keras.utils.Sequence):
    def __init__(self, X, y, batch_size, preprocess: Callable = None, *args, **kwargs):
        super().__init__()
        self._X = X
        self._y = y
        self._preprocess_fn = preprocess
        self._batch_size = batch_size
        
    def _preprocess(self, X, y):
        if self._preprocess_fn is not None:
            return self._preprocess_fn(X, y)
        return (X / 255.).reshape((-1, 28*28)), y
        
    def __len__(self):
        return int(np.ceil(len(self._X) / float(self._batch_size)))
    
    def __getitem__(self, idx):
        batch_idx = slice(idx*self._batch_size, (idx+1)*self._batch_size, 1)
        x_batch = self._X[batch_idx]  # self._X[idx*self._batch_size:(idx+1)*self._batch_size]
        y_batch = self._y[batch_idx]
        return self._preprocess(x_batch, y_batch)
    
    
class BatchNormFlawed:
    def __init__(self, name):
        self.trainable = True
        self._beta = tf.Variable(0, dtype='float64')
        self._gamma = tf.Variable(1,  dtype='float64')
        self.name = name
        
    def __call__(self, x, writer=None, step=None):
        mu = tf.reduce_mean(x, axis=0)
        sigma = tf.math.reduce_std(x, axis=0)
        normed = (x - mu) / sigma 
        out = normed * self._gamma + self._beta
        
        if writer is not None:
            with writer.as_default():
                tf.summary.histogram(self.name + '_beta', self._beta, step=step)
                tf.summary.histogram(self.name + '_gamma', self._gamma, step=step)
                tf.summary.histogram(self.name + '_normed', normed, step=step)
                tf.summary.histogram(self.name + '_out', out, step=step)
                tf.summary.histogram(self.name + '_sigma', sigma, step=step)
                tf.summary.histogram(self.name + '_mu', mu, step=step)

        return out
    
    def get_trainable(self):
        if self.trainable: 
            return [self._beta, self._gamma]
        else:
            return []

        
from typing import Callable


class Dense:
    def __init__(self, inp_shape, out_shape, activation: Callable, name):
        self.trainable = True
        self._inp_shape = inp_shape
        self._out_shape = out_shape
        self._activation = activation
        
        self._w = tf.Variable(np.random.randn(inp_shape, out_shape))
        self._b = tf.Variable(np.zeros((1, out_shape)))
        
        self.name = name
        
    def __call__(self, x, writer=None, step=None):
        val = x @ self._w + self._b
        a = self._activation(val)
        if writer is not None:
            with writer.as_default():
                tf.summary.histogram(self.name + '_kernel', self._w, step=step)
                tf.summary.histogram(self.name + '_bias', self._b, step=step)
                tf.summary.histogram(self.name + '_activation', a, step=step)
                tf.summary.histogram(self.name + '_z', val, step=step)
        return a
    
    def get_trainable(self):
        if self.trainable: 
            return [self._w, self._b]
        else:
            return []
        
    @property
    def inp_shape(self):
        return self._inp_shape
    
    @property
    def out_shape(self):
        return self._out_shape
    
    @property
    def w(self):
        return self._w
    
    @property
    def b(self):
        return self._b
    
    
class DenseSmart:
    def __init__(self, inp_shape, out_shape, activation: Callable, name):
        self.trainable = True
        self._inp_shape = inp_shape
        self._out_shape = out_shape
        self._activation = activation
        
        if 'sigmoid' in self._activation.__name__:
            self._w = tf.Variable(np.random.rand(inp_shape, out_shape) * np.sqrt(6 / (inp_shape + out_shape)))
        elif 'relu' in self._activation.__name__:
            self._w = tf.Variable(np.random.randn(inp_shape, out_shape) * np.sqrt(2 / (inp_shape)))
        else:
            # Just a Normal
            self._w = tf.Variable(np.random.randn(inp_shape, out_shape))      
        self._b = tf.Variable(np.zeros((1, out_shape)))
        self.name = name
        
    def __call__(self, x, writer=None, step=None):
        val = x @ self._w + self._b
        a = self._activation(val)
        if writer is not None:
            with writer.as_default():
                tf.summary.histogram(self.name + '_kernel', self._w, step=step)
                tf.summary.histogram(self.name + '_bias', self._b, step=step)
                tf.summary.histogram(self.name + '_activation', a, step=step)
                tf.summary.histogram(self.name + '_z', val, step=step)
        return a
    
    def get_trainable(self):
        if self.trainable: 
            return [self._w, self._b]
        else:
            return []
        
    @property
    def inp_shape(self):
        return self._inp_shape
    
    @property
    def out_shape(self):
        return self._out_shape
    
    @property
    def w(self):
        return self._w
    
    @property
    def b(self):
        return self._b
    
    
class Sequential:
    def __init__(self, *args):
        self._layers = args
        self._trainable_variables = [i for s in [l.get_trainable() for l in self._layers] for i in s] 
        
    def _forward(self, x, writer=None, step=None):
        for layer in self._layers:
            x = layer(x, writer, step)
        return x
        
    def fit_generator(self, train_seq, eval_seq, epoch, loss, optimizer, writer=None):
        history = dict(train=list(), val=list())
        
        train_loss_results = list()
        val_loss_results = list()

        train_accuracy_results = list()
        val_accuracy_results = list()
        
        step = 0
        for e in range(epoch):
            p = tf.keras.metrics.Mean()
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_loss_avg_val = tf.keras.metrics.Mean()

            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            epoch_accuracy_val = tf.keras.metrics.SparseCategoricalAccuracy()

            for x, y in train_seq:
                with tf.GradientTape() as tape:
                    prediction = self._forward(x, writer, step)
                    loss_value = loss(y, prediction)
                gradients = tape.gradient(loss_value, self._trainable_variables)
                optimizer.apply_gradients(zip(gradients, self._trainable_variables))
                epoch_accuracy.update_state(y, prediction)
                epoch_loss_avg.update_state(loss_value)
                
                with writer.as_default():
                    tf.summary.scalar('train_accuracy', epoch_accuracy.result().numpy(), step=step)
                    tf.summary.scalar('train_loss', epoch_loss_avg.result().numpy(), step=step)

                step += 1
                
            train_accuracy_results.append(epoch_accuracy.result().numpy())
            train_loss_results.append(epoch_loss_avg.result().numpy())


            for x, y in eval_seq:
                prediction = self._forward(x)
                loss_value = loss(y, prediction)
                epoch_loss_avg_val.update_state(loss_value)
                epoch_accuracy_val.update_state(y, prediction)
            
            val_accuracy_results.append(epoch_accuracy_val.result().numpy())
            val_loss_results.append(epoch_loss_avg_val.result().numpy())

            # print(f"Epoch train loss: {epoch_train_loss[-1]:.2f},\nEpoch val loss: {epoch_val_loss[-1]:.2f}\n{'-'*20}")
            print("Epoch {}: Train loss: {:.3f} Train Accuracy: {:.3f}".format(e + 1,
                                                                               train_loss_results[-1],
                                                                               train_accuracy_results[-1]))
            print("Epoch {}: Val loss: {:.3f} Val Accuracy: {:.3f}".format(e + 1,
                                                                           val_loss_results[-1],
                                                                           val_accuracy_results[-1]))
            print('*' * 20)

        return None
            
    def predict_generator(self, seq):
        predictions = list()
        for x in seq:
            predictions.append(self._forward(x).numpy())
        return np.vstack(predictions)
    
    @property
    def trainable_variables(self):
        return self._trainable_variables