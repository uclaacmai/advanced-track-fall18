from TwoLayerNet import NeuralNetwork
import numpy as np
from load_mnist import MNIST_Loader



if __name__ == '__main__':
    a = MNIST_Loader()
    X_train, y_train = a.load_mnist('../data')
    X_test, y_test = a.load_mnist('../data', 't10k')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
    net = NeuralNetwork(input_size=X_train.shape[1], hidden_size=50, output_size=10)
    train_dict = net.train(X=X_train, y=y_train, X_val=X_test, y_val=y_test, num_iters=1000, verbose=True)
    print('Final loss: {}, final training accuracy: {} final test accuracy: {}'.format(train_dict['loss_history'][-1], train_dict['train_acc_history'][-1], train_dict['val_acc_history'][-1]))
