from TwoLayerNet import NeuralNetwork
import numpy as np
from keras.datasets import mnist

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[1]))
    print('X train shape: {} y train shape : {} X test shape: {} Y test shape: {}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    net = NeuralNetwork(input_size=X_train.shape[1], hidden_size=50, output_size=10)
    train_dict = net.train(X=X_train, y=y_train, X_val=X_test, y_val=y_test, num_iters=1000, verbose=True)
    print('Final loss: {}, final training accuracy: {} final test accuracy: {}'.format(train_dict['loss_history'][-1], train_dict['train_acc_history'][-1], train_dict['val_acc_history'][-1]))
