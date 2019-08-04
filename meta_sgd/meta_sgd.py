import numpy as np
from utils import data_points_sample


class MetaSGD(object):

    def __init__(self, num_tasks=2, num_samples=10, epochs=10000, alpha=.0001, beta=.0001):
        self.num_tasks = num_tasks
        self.num_samples = num_samples
        self.epochs = epochs

        self.alpha = alpha
        self.beta = beta

        self.theta = np.random.normal(size=50).reshape(50, 1)
        self.alpha = np.random.normal(size=50).reshape(50, 1)

    def sigmoid(self, a):
        return 1.0 / (1 + np.exp(-a))

    def train(self):
        self.theta_ = []

        for e in range(self.epochs):
            loss = self.update_theta_loss(self.theta_)
            meta_gradient = self.update_meta_gradient(self.theta_)

            self.theta = self.theta - self.beta * meta_gradient / self.num_tasks
            self.alpha = self.alpha - self.beta * meta_gradient / self.num_tasks

            if e % 1000 == 0:
                print("Epoch {}: Loss {}\n".format(e, loss))
                print('Updated Model Parameter Theta\n')
                print('Sampling Next Batch of Tasks \n')
                print('---------------------------------\n')

    def update_theta_loss(self, theta_):
        for i in range(self.num_tasks):
            x_train, y_train = data_points_sample(self.num_samples)
            a = np.matmul(x_train, self.theta)
            y_hat = self.sigmoid(a)

            g_sum = np.matmul(x_train.T, (y_hat - y_train))
            gradient = g_sum / self.num_samples

            theta_i = self.theta - (np.multiply(self.alpha, gradient))
            theta_.append(theta_i)

            y_hat_mat = np.matmul(-y_train.T, np.log(y_hat))
            diff_mat = np.matmul((1 - y_train.T), np.log(1 - y_hat))
            l_sum = y_hat_mat - diff_mat
            loss = (l_sum / self.num_samples)[0][0]

        return loss

    def update_meta_gradient(self, theta_):
        meta_gradient = np.zeros(self.theta.shape)

        for i in range(self.num_tasks):
            x_test, y_test = data_points_sample(10)
            a = np.matmul(x_test, self.theta_[i])
            y_pred = self.sigmoid(a)

            mat_sum = np.matmul(x_test.T, (y_pred - y_test))
            meta_gradient += mat_sum / self.num_samples

        return meta_gradient
