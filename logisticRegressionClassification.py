from __future__ import division
'''
This implements a multi-class logistic regression model to recognize
handwritten digits (form 0 to 9)
'''

import numpy as np


np.random.seed(12345)


def prepare_data(traindatafile, split=0.75):
    """
    :param traindatafile: the file from which to load the to train
    and test the model on.
    :param split: proportion of data in the training
    :return: a list containg the training set and the validation set
    """
    # skip the header row
    data = np.genfromtxt(traindatafile, delimiter=',', skip_header=1)
    #
    subset = np.random.rand(np.shape(data)[0]) < split
    train = data[subset]
    ytrain = np.copy(train[:, 0])

    # Normalize by data as X = (X - min)/ (max - min)
    # min = 0, and max = 255, so X = X/255.
    train = train*(1.0/255)
    # Add the intercept term
    train[:, 0] = 1
    test = data[~subset]
    ytest = np.copy(test[:, 0])
    test = test*(1.0/255)
    # Add the intercept term to the training set
    test[:, 0] = 1

    return train, ytrain, test, ytest


def sigmoid(z):
    """
    J = SIGMOID(z) Compute the sigmoid of z
    :param z: a vector
    :return: sigmoid of z
    """
    return 1.0/(1.0 + np.exp(-z))


class LogisticModel(object):

    def __init__(self):
        self.lambdareg = 0.01

    def cost_function(self, theta, X, y, lambdareg):
        """
        :param theta: the parameter vector theta
        :param X: A matrix containing input training data
        :param y: the response variable y
        :param lambdareg: the regularization term lambda
        :return: the cost function J and the gradien grad
        """
        # number of training examples.
        m = np.shape(y)[0]

        # Get as many rows as in theta
        grad = np.zeros([np.shape(theta)[0], 1])
        J = 0
        for k in range(0, m):
            # Get row k of the ndarray X

            z = np.dot(np.transpose(theta), np.transpose(X[k, : ]))

            J += (-1.0/m) * (y[k] * np.log(sigmoid(z)) + (1 - y[k])*np.log(1 - sigmoid(z)))

        regterm = 0
        # Do not apply regularization to the intercept term

        for r in range(1, np.shape(X)[1]):
            regterm += (lambdareg/2)*np.power(theta[r], 2)

        # Let us compute the gradient

        J += regterm

        # Intercept term
        for k in range(0, m):

            z = np.dot(np.transpose(theta), np.transpose(X[k, :]))

            grad[0] += ((1/m) * (sigmoid(z) - y[k])*X[k, 0])

        for r in range(1, np.shape(X)[1]):
            for k in range(0, m):
                z = np.dot(np.transpose(theta), np.transpose(X[k, :]))
                grad[r] += (1.0/m) * ((sigmoid(z) - y[k])*X[k, r])

            grad[r] += (lambdareg/m)*theta[r]

        print "Cost function J: ", J

        return J, grad

    def gradientdescent(self, X, y, initial_theta, lambdareg, alpha=0.01, number_iterations=50):

        new_theta = initial_theta
        # A column vector of length number_iterations
        cost_history = np.zeros([number_iterations, 1])

        for k in range(number_iterations):
            cost, grad = self.cost_function(theta=new_theta, X=X, y=y,
                                           lambdareg=lambdareg)

            new_theta = np.subtract(new_theta, alpha*grad)
            cost_history[k] = cost
        return new_theta, cost_history

    def train(self, train, ytrain):
        """
        :param train: A matrix containing data to train on
        :param ytrain:
        :return: trained theta along with the successibe values of the
        cost function J.
        """

        initial_theta = np.zeros([np.shape(train)[1], 1])

        theta, J_history = self.gradientdescent(train, ytrain,
                                                initial_theta=initial_theta, lambdareg=1)

        return theta, J_history
    
    def predict(self, theta, newdata):
        """
        :param theta: a matrix containing theta learned from training data
        rows correspond to parameters for each model! from 0 - 9
        For example, the first row contains thetas for the model that predicts 0
        :param newdata: a matrix containming new data
        :return: predicted probabilities 
        """
        def probs_func(x):
            return max(x), np.argmax(x)

        probs = np.zeros([np.shape(newdata)[0], 1])

        # Matrix for predictions one for each model!

        probs_matrix = np.zeros([np.shape(newdata)[0], 10])

        m = np.shape(newdata)[0]

        nber_classes = np.shape(theta)[0]

        predictions = np.zeros([np.shape(newdata)[0], 2])

        for k in range(nber_classes):
            for j in range(m):
                z = np.dot(theta[k, :], np.transpose(newdata[j, :]))
                probs[j] = sigmoid(z)

            probs_matrix[:, k] = np.transpose(probs)

        predictions = np.apply_along_axis(probs_func, 1, probs_matrix)

        return predictions
        

def main():

    logit_model = LogisticModel()

    classes = 10
    #datafilename = "train.csv"
    datafilename = "copyfile.csv"

    train, ytrain, test, ytest = prepare_data(datafilename, split=0.80)

    # We will use all vs one; however, for testing, we just need to test a binary
    # classifier. So, we are trying to predict zeros

    thetas_matrix = np.zeros([classes, np.shape(train)[1]])

    for k in range(classes):
        newytrain = [int(y == k) for y in ytrain]

        theta, J_history = logit_model.train(train, newytrain)

        thetas_matrix[k, :] = np.transpose(theta)
        print "This is model ", k,

    preds = logit_model.predict(thetas_matrix, test)

    total = 0.0
    for k in range(np.shape(preds)[0]):
        if preds[k,1] == ytest[k]:
            total += 1

    # Weighted empirical error
    print "Accuracy: ", float(total)/np.shape(preds)[0]


if __name__ == '__main__':
    main()


