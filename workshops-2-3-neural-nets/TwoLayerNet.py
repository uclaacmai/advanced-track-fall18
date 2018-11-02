import numpy as np
import matplotlib.pyplot as plt

"""Credits: This NeuralNetwork class was adapted from Stanford's CS 231N class: http://cs231n.stanford.edu/"""
class NeuralNetwork(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.
  In other words, the network has the following architecture:
  input - fully connected layer - ReLU - fully connected layer - softmax
  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:
    W1: First layer weights; has shape (H, D)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (C, H)
    b2: Second layer biases; has shape (C,)
    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}

    # TODO: Initialize the weights and biases according to the shapes above
    self.params['W1'] = None
    self.params['b1'] = None
    self.params['W2'] = None
    self.params['b2'] = None

    self.params['W1'] = std * np.random.randn(hidden_size, input_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(output_size, hidden_size)
    self.params['b2'] = np.zeros(output_size)


  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.
    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.
    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].
    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    # Compute the forward pass

    z1 = None # TODO: compute the result of the forward through the first layer of the network (before the nonlinearity is applied.)
    h1 = None # TODO: apply the RELU nonlinearity to the above result.
    z2 = None # TODO: compute the result of passing h1 through the next layer in the network.
    scores = z2

    z1 = X.dot(W1.T) + b1
    h1 = z1 * (z1 > 0)
    z2 = h1.dot(W2.T) + b2
    scores = z2

  
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # scores is num_examples by num_classes

    # first, normalize the scores
    normscores = scores - np.max(scores, axis = 1, keepdims = True)
    # now calculate the softmax function
    score_probs = np.exp(normscores)
    score_probs/=np.sum(score_probs, axis = 1, keepdims = True)
    # now, pick out the correct ones, sum, and norm
    d_loss = -np.sum(np.log(score_probs[np.arange(scores.shape[0]), y].clip(min=np.finfo(float).eps))) / scores.shape[0]
    # penalize by frobenius norm
    r_loss = None # TODO: Write out the contribution that L2-regularization makes to the overall loss function.
    r_loss = .5 * reg * (np.sum(W1 **2) + np.sum(W2 ** 2))
    loss = None # TODO: Compute the overall loss as the regularized cross-entropy loss.
    loss = d_loss + r_loss


    grads = {}

    # ================================================================ #
    # TODO
    #   Implement the backward pass.  Compute the derivatives of the 
    #   weights and the biases.  Store the results in the grads
    #   dictionary.  e.g., grads['W1'] should store the gradient for 
    #   W1, and be of the same size as W1.
    # ================================================================ #

    # calculate the grad with respect to softmax
    p = score_probs.copy()
    # to account for the case where w_j = w_{y_i} (i.e. our class corresponds to the correct class label)
    p[range(X.shape[0]),y]-=1
    p/=X.shape[0]
    # Given p, the incoming gradient into the 2nd layer of the network, implement backprop for the rest of the network's weights and biases.
    # now backprop into the bias
    # the second layer is h_2 = W_2(h_1) + b_2
    # so for the bias, given the gradient, what is the gradient with respect to the bias?
    dldb2 = None # TODO: Implement backpropagation into the biases for the 2nd layer.
    dldb2 = np.sum(p, axis = 0)
    # now backprop to the second layer weights
    # since we computed Wh1 where h1 was the input into this layer, derivative is h1, and add derivative of regularization func
    dldw2 = None # TODO: Implement backpropagation into the weights for the 2nd layer.
    dldw2 = p.T.dot(h1) + reg * W2
    # calculate the gradient that we send back
    # basically this is the gradient of the inputs into this layer, h1
    # since we did Wh1 the grad is just W, times p  for the chain rule
    dLdh1 = None # TODO: this is the "upstream gradient" for the first layer, where as p was the upstream for second layer
    dLdh1 = p.dot(W2)
    # we have h_1 = relu(W_1x + b_1)
    # now back into the relu.
    dldz = None # TODO: backprop into the relu
    dldz = dLdh1 * (z1 > 0)
    # now back into the first layer bias
    dldb1 = None # TODO: Implement backprop into the first layer biases.
    dldb1 = np.sum(dldz, axis = 0)
    # now back into the first layer weights
    dldw1 = None # TODO: implement backprop into the first layer weights
    dldw1 = dldz.T.dot(X) + reg * W1

    # assign grads
    grads['b2'] = dldb2
    grads['W2'] = dldw2
    grads['b1'] = dldb1
    grads['W1'] = dldw1
    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=100, verbose=False):
    """
    Train this neural network using stochastic gradient descent.
    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in np.arange(num_iters):
      batch_indices = None # TODO: create a minibatch by sampling indices here.
      batch_indices = np.random.choice(X.shape[0], batch_size)
      X_batch = X[batch_indices]
      y_batch = y[batch_indices]


       # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      # TODO: perform a gradient descent update for all the parameters.
      # self.params['b1'] = None
      # self.params['W1'] = None
      # self.params['b2'] = None
      # self.params['W2'] = None


      self.params['b1'] = self.params['b1']-learning_rate*grads['b1']
      self.params['W1'] = self.params['W1']-learning_rate*grads['W1']
      self.params['b2'] = self.params['b2']-learning_rate*grads['b2']
      self.params['W2'] = self.params['W2']-learning_rate*grads['W2']

      if verbose and it % 100 == 0:
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        print('iteration {} / {}: loss {} train acc {} val acc {}'.format(it, num_iters, loss, train_acc, val_acc))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.
    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.
    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None
    #TODO: implement code to predict the label given the input features.
    # get the scores
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    z1 = X.dot(W1.T) + b1
    h1 = z1 * (z1 > 0)
    z2 = h1.dot(W2.T) + b2
    scores = z2
    y_pred = np.argmax(scores, axis = 1)
    return y_pred