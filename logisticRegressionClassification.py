from __future__ import division
'''
This implements a multi-class logistic regression model to recognize
handwritten digits (form 0 to 9)
'''

import numpy as np
from multiprocessing import Process, Queue

import multiprocessing

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

    np.random.seed(12345)
    # Randomly shuffle the data set
    np.random.shuffle(data)

    np.random.seed(12345)
    subset = np.random.rand(np.shape(data)[0]) < split
    train = data[subset]
    ytrain = np.copy(train[:, 0])

    # Normalize by data as X = (X - min)/ (max - min)
    # min = 0, and max = 255, so X = X/255.
    train *= 1.0/255 # train = train*(1.0 / 255)
    # Add the intercept term
    train[:, 0] = 1
    test = data[~subset]
    ytest = np.copy(test[:, 0])
    test *= 1.0 / 255 #  test*(1.0/255)
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


def makeChunks(a, n):
    """
    split the list a into chuncks of maximum size n
    :param a: the list split into chunks
    :param n: the size of each chunk
    :return: Yield successive chunks of size n from a.
    """
    for k in range(0, len(a), n):
        yield a[k:k+n]


class LogisticModel(object):

    def __init__(self):
        self.lambdareg = 0.01

    def cost_function(self, theta, X, y, lambda_reg, train_size=None):
        """
        :param theta: the parameter vector theta
        :param X: A matrix containing input training data
        :param y: the response variable y
        :param lambda_reg: the regularization term lambda
        :param train_size: the overall training set size. We need this parameter to be set
        in order to use a this cost function in the parallel gradient descent function.
        defaults to None, meaning that by default it is used with batch gradient descent
        :return: the cost function J and the gradien grad
        """

        # number of training examples.
        m = np.shape(y)[0]

        if train_size is not None:
            size_train_set = train_size
        else:
            size_train_set = m

        # Get as many rows as in theta
        grad = np.zeros([np.shape(theta)[0], 1])
        J = 0
        for k in range(0, m):
            # Get row k of the ndarray X

            z = np.dot(np.transpose(theta), np.transpose(X[k, : ]))

            J += (-1.0/size_train_set) * (y[k] * np.log(sigmoid(z)) + (1 - y[k])*np.log(1 - sigmoid(z)))

        regterm = 0
        # Do not apply regularization to the intercept term

        for r in range(1, np.shape(X)[1]):
            regterm += (lambda_reg/2)*np.power(theta[r], 2)

        # Let us compute the gradient

        J += regterm

        # Intercept term
        for k in range(0, m):

            z = np.dot(np.transpose(theta), np.transpose(X[k, :]))

            grad[0] += ((1/size_train_set) * (sigmoid(z) - y[k])*X[k, 0])

        for r in range(1, np.shape(X)[1]):
            for k in range(0, m):
                z = np.dot(np.transpose(theta), np.transpose(X[k, :]))
                grad[r] += (1.0/size_train_set) * ((sigmoid(z) - y[k])*X[k, r])

            grad[r] += (lambda_reg/size_train_set)*theta[r]

        #print "Cost function J: ", J

        return J, grad

    def gradientDescent(self, X, y, initial_theta, do_output=None, lambdareg=1, alpha=0.001, number_iterations=50):
        """
        :param X: A matrix containing independent variables to train on
        :param y: the dependent variable
        :param initial_theta: initial values of theta
        :param lambdareg: The regularization parameter lambda; default to 1
        :param alpha: the learning rate alpha; default to 0.001
        :param number_iterations: number of gradient decent iterations
        :param do_output: if provided, this do_output is a 2-length list containing the number
        of iterations after which to output as first element of the list and the output file name as the second
        :return: the value of theta leraned from the data and the associated cost function
        """

        new_theta = initial_theta
        # A column vector of length number_iterations
        cost_history = np.zeros([number_iterations, 1])

        for k in range(number_iterations):
            cost, grad = self.cost_function(theta=new_theta, X=X, y=y,
                                            lambda_reg=lambdareg)

            new_theta = np.subtract(new_theta, alpha*grad)

            cost_history[k] = cost

            if do_output is not None and int(do_output[0]) == k:
                np.savetxt(do_output[1], cost_history, delimiter=",")

        return new_theta, cost_history

    def DoSum(self, X, y, nb_cores, theta, worker, queue, lambda_reg=1):
        """
        DoSum is very similar to the cost function implemented earlier except that the average over
        the training set is omitted
        This could well be the "Reduce" step of a Map-Reduce program
        :param X: the entire training set
        :param y: the dependent variable (DV)
        :param nb_cores: the number of CPU cores on this computer
        :param initial_theta: a vector containing the current value of theta
        :param worker: the number of the current process
        :param queue: A queue in which the current process puts its result
        :param lambda_reg: the regularization parameter
        :return: Nothing
        """

        split = int(np.shape(X)[0]/nb_cores)
        remaining = np.shape(X)[0] % split
        start = worker * split
        end = split*(worker + 1)
        # the last worker gets whatever is left. Workers are referenced from 0
        if worker == nb_cores - 1:
            end += remaining

        # Use the train_size argument to divide over the entire training set size
        cost, grad = self.cost_function(theta=theta, X=X[start:end:], y=y[start:end:],
                                        lambda_reg=lambda_reg, train_size=np.shape(X)[0])

        queue.put({"grad": grad, "cost": cost})

    def GradientDescentParallel(self, train, ytrain, initial_theta, lambda_reg=1, alpha=0.01, nbr_iterations=50):
        """
        This implementation of Gradient descent using a Map-Reduce like approach, will split the input
        data set into the number of cores available on the system and compute sum on each of them
        :param train:
        :param ytrain:
        :param initial_theta:
        :param lambda_reg:
        :param alpha:
        :param nbr_iterations:
        :return:
        """
        # Get the number of CPU cores available on the system
        cpu_cores = multiprocessing.cpu_count()
        cost_history = np.zeros([nbr_iterations, 1])
        new_theta = initial_theta

        queue = Queue()
        for iteration in range(nbr_iterations):
            workers_list = []

            for worker in range(cpu_cores):
                p = Process(target=self.DoSum, args=(train, ytrain, cpu_cores, new_theta, worker, queue, lambda_reg, ))
                p.start()
                workers_list.append(p)

            for p in workers_list:
                p.join()
            cost = 0
            sub_sum = np.zeros([np.shape(train)[1], 1])
            while not queue.empty():
                res = queue.get()
                sub_sum = np.add(sub_sum, res["grad"])
                cost += res["cost"]

            new_theta = np.subtract(new_theta, alpha * sub_sum)
            cost_history[iteration] = cost
            print "Cost function: ", cost
        return new_theta, cost_history

    def train(self, train, ytrain):
        """
        :param train: A matrix containing data to train on
        :param ytrain: the dependent variable
        :return: trained theta along with the successibe values of the
        cost function J.
        """

        initial_theta = np.zeros([np.shape(train)[1], 1])

        #theta, J_history = self.gradientDescent(train, ytrain,
        #                                        initial_theta=initial_theta, lambdareg=1)

        theta, J_history = self.GradientDescentParallel(train, ytrain,
                                                        initial_theta=initial_theta, lambda_reg=1)

        return theta, J_history

    def doWork(self, train, ytrain, queue, workerid, iterations=50):
        """
        :param train: A matrix containing data to train on
        :param ytrain: the dependent variable
        :param queue: A queue in which the worker puts its result
        :param workerid: the worker's id
        :param iterations: number of iterations after which the worker
        writes its cost function to a text file. The file's names is the
        worker's id
        :return: Nothing
        """
        jcost_filename = "worker_%d.csv" % workerid
        do_output = [iterations, jcost_filename]

        # convert the problem into a binary classification problem
        newytrain = [int(y == workerid) for y in ytrain]

        initial_theta = np.zeros([np.shape(train)[1], 1])

        theta, J_history = self.gradientDescent(train, newytrain, initial_theta=initial_theta, do_output=do_output)

        trained_params = {'theta': theta, 'worker_id': workerid, 'j_cost': J_history}
        # Save the cost function history
        queue.put(trained_params)
        np.savetxt(jcost_filename, trained_params['j_cost'], delimiter=",")

    def trainParallel(self, train, ytrain, workers=10):
        """
        :param train: train data, the response variable is not included
        :param ytrain: dependent or response variable
        :param workers: the number of workers, typically the number of classes
        :return: a matrix in which each row contains trained thetas for the corresponding class, for example,
         the first row contains theta for the first class. that is response variable 0.
        """

        thetas_matrix = np.zeros([workers, np.shape(train)[1]])

        # Get the nuimber of available cores!!
        cpu_cores = multiprocessing.cpu_count()
        # split workers into available cores
        chunks = makeChunks(range(workers), cpu_cores)

        queue = Queue()
        for sub_list in chunks:
            workers_list = []
            for worker_id in sub_list:
                p = Process(target=self.doWork, args=(train, ytrain, queue, worker_id, ))
                p.start()
                workers_list.append(p)

            for p in workers_list:
                p.join()

            while not queue.empty():
                d = queue.get()
                k = int(d['worker_id'])
                thetas_matrix[k, :] = np.transpose(d['theta'])

        return thetas_matrix
    
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
    datafilename = "devfile.csv"
    #datafilename = "copyfile.csv"

    train, ytrain, test, ytest = prepare_data(datafilename, split=0.80)

    # We will use all vs one; however, for testing, we just need to test a binary
    # classifier. So, we are trying to predict zeros

    """
    Example call of the parallel fit

    thetas_matrix = logit_model.trainParallel(train, ytrain, workers=classes)

    """

    """
    Example call of the sequential implementation
    This however uses GradientDescentParallel, which splits the data into multiple chunks
    and perform gradient descent on each of them in parallel
    """

    thetas_matrix = np.zeros([classes, np.shape(train)[1]])

    for k in range(classes):
        newytrain = [int(y == k) for y in ytrain]

        theta, J_history = logit_model.train(train, newytrain)

        thetas_matrix[k, :] = np.transpose(theta)
        #print "This is model ", k,

    #print thetas_matrix

    preds = logit_model.predict(thetas_matrix, test)

    total = 0.0
    for k in range(np.shape(preds)[0]):
        if preds[k,1] == ytest[k]:
            total += 1

    # Weighted empirical error, we can use it as an estimate of the accuracy of our algorithm
    # because, classes are more or less evenly distributed
    print "Accuracy based on the empirical error: ", float(total)/np.shape(preds)[0]


if __name__ == '__main__':
    main()


