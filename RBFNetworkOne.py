import csv
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans


input_data = []
input_points = []
output_labels = []


# Create a test and train split

train_data = pd.read_csv(
    'D:\\UniversityOfWaterloo\\Courses\\657_ToolsOfIntelligentSystemDesign\\Assignment2\\ece657assignment2\\Problem3\\RBF_data.csv').values

# print(len(train_data))
split_val = round(0.8*len(train_data))
indices = np.random.permutation(train_data.shape[0])

training_idx, test_idx = indices[:split_val], indices[split_val:]
training_data, testing_data = train_data[training_idx,
                                         :], train_data[test_idx, :]

# No fo random points
no_rand_points = 150
random.seed(3)
# Part 1 : Random points
pos = np.random.permutation(len(training_data))[:no_rand_points]
#random_data = training_data[pos]

# random_data_points = np.random.randint(
# 0, training_data.shape[0], no_rand_points)
random_data = training_data[pos]


# Part 2 : KMeans clustering
random_kmeans_data = KMeans(n_clusters=no_rand_points, random_state=0).fit(
    training_data).cluster_centers_


# Generate data as per requirements
def GenerateData(exportData):
    label = 0
    for a in range(0, 21):
        for b in range(0, 21):
            xi = -2 + (0.2 * a)
            xj = -2 + (0.2 * b)
            input_data.append([round(xi, 2), round(xj, 2)])
            input_points.append([a, b])
            if(((xi**2)+(xj**2)) <= 1):
                output_labels.append(1)
            if(((xi**2)+(xj**2)) > 1):
                output_labels.append(-1)
            label = label + 1
    if(exportData == True):
        with open('RBF_data.csv', 'w', newline='') as f:
            #header = ['i', 'j', 'xi', 'xj', 'output']
            writer = csv.writer(f,  delimiter=',')
            # writer.writerow(header)
            for row in zip(input_points, input_data, output_labels):
                data = [row[0][0], row[0][1], row[1][0], row[1][1], row[2]]
                writer.writerow(data)


class RBFNetwork:

    def __init__(self, inputData, hiddenData, sigma):

        self.input_data = inputData[:, 2:4]
        self.hidden_layer_data = hiddenData[:, 2:4]
        self.output_labels = inputData[:, 4]

        self.input_layer = self.input_data.shape[0]
        self.hidden_layer = self.hidden_layer_data.shape[0]
        # self.output_layer =

        self.sigma = sigma
        self.matrix = np.zeros([self.input_layer, self.hidden_layer])
        self.weights = np.zeros(hiddenData.shape[0])

        self.phi_matrix()
        self.pseudo_inverse()
        self.calculate_weight()

        pass

    def phi(self, input, target):
        phi_value = np.exp(-((input[0]-target[0])**2 +
                           (input[1]-target[1])**2)/(2*self.sigma**2))
        return phi_value

    def phi_matrix(self):
        self.matrix = np.zeros(
            [self.input_data.shape[0], self.hidden_layer_data.shape[0]])
        for a in range(0, self.input_data.shape[0]):
            for b in range(0, self.hidden_layer_data.shape[0]):
                self.matrix[a][b] = self.phi(
                    self.input_data[a], self.hidden_layer_data[b])

    def pseudo_inverse(self):
        mtm = np.dot(np.transpose(self.matrix), self.matrix)
        pi_matrix = np.linalg.inv(mtm)
        self.p_inverse = np.dot(pi_matrix, np.transpose(self.matrix))
        pass

    def calculate_weight(self):
        self.weights = np.dot(self.p_inverse, self.output_labels)
        pass

    def predicted_data(self, testData, errorCheck):
        if(errorCheck == True):
            self.input_data = testData[:, 2:4]
            self.phi_matrix()
            output_predicted = np.dot(self.matrix, self.weights)
        else:
            output_predicted = np.dot(self.matrix, self.weights)
        return output_predicted

    def mean_squared_error(self, predictedOutputData, actualOutputData, testVal):
        if (testVal == True):
            mse_error = np.square(np.subtract(
                actualOutputData, predictedOutputData)).mean()
        else:
            mse_error = np.square(np.subtract(
                actualOutputData, predictedOutputData[0:88])).mean()
        return mse_error


def main():

    GenerateData(False)
    sigma_list = []
    mse_train_list = []
    mse_test_list = []

    mse_train_random_list = []
    mse_test_random_list = []

    mse_train_kmeans_list = []
    mse_test_kmeans_list = []

    a = 0.2
    while (a < 4.0):
        sigma_list.append(round(a, 2))

        # Part 1 : Hidden layer cluster is equal to number of inputs
        RBF_model = RBFNetwork(training_data, training_data, a)
        predicted_train_op = RBF_model.predicted_data(testing_data, False)
        mse_train_error = RBF_model.mean_squared_error(
            predicted_train_op, training_data[:, 4], True)
        predicted_test_op = RBF_model.predicted_data(testing_data, True)
        mse_test_error = RBF_model.mean_squared_error(
            predicted_test_op, testing_data[:, 4], False)
        mse_train_list.append(round(mse_train_error, 3))
        mse_test_list.append(round(mse_test_error, 3))
        print("##########################################")
        print("Part 1 :")
        print("Sigma = " + str(round(a, 2)))

        print("Training data MSE: " + str(mse_train_error))
        print("Testing data MSE: " + str(mse_test_error))
        print("##########################################")
        mse_train_random_error = 0
        mse_test_random_error = 0
        # Part 2 : Hidden layer is random number of 150 input
        RBF_model_random = RBFNetwork(training_data, random_data, a)

        predicted_train_rand_op = RBF_model_random.predicted_data(
            testing_data, False)
        mse_train_random_error = RBF_model_random.mean_squared_error(
            predicted_train_rand_op, training_data[:, 4], True)
        predicted_test_rand_op = RBF_model_random.predicted_data(
            testing_data, True)
        mse_test_random_error = RBF_model_random.mean_squared_error(
            predicted_test_rand_op, testing_data[:, 4], False)
        mse_train_random_list.append(round(mse_train_random_error, 3))
        mse_test_random_list.append(round(mse_test_random_error, 3))

        print("Part 2 :")
        print("Sigma = " + str(round(a, 2)))

        print("Training data MSE: " + str(mse_train_random_error))
        print("Testing data MSE: " + str(mse_test_random_error))
        print("##########################################")
        mse_train_kmeans_error = 0
        mse_test_kmeans_error = 0
        # Part 3 : Hidden layers is decided by the k-means clustering algorithm
        RBF_model_kmeans = RBFNetwork(training_data, random_kmeans_data, a)

        predicted_train_kmeans_op = RBF_model_kmeans.predicted_data(
            testing_data, False)
        mse_train_kmeans_error = RBF_model_kmeans.mean_squared_error(
            predicted_train_kmeans_op, training_data[:, 4], True)
        predicted_test_kmeans_op = RBF_model_kmeans.predicted_data(
            testing_data, True)
        mse_test_kmeans_error = RBF_model_kmeans.mean_squared_error(
            predicted_test_kmeans_op, testing_data[:, 4], False)
        mse_train_kmeans_list.append(round(mse_train_kmeans_error, 3))
        mse_test_kmeans_list.append(round(mse_test_kmeans_error, 3))

        print("Part 3 :")
        print("Sigma = " + str(round(a, 2)))
        a = a + 0.2

        print("Training data MSE: " + str(mse_train_kmeans_error))
        print("Testing data MSE: " + str(mse_test_kmeans_error))
        print("##########################################")

    with open('RBF_methodOne_output.csv', 'w', newline='') as f:
        writer_op = csv.writer(f,  delimiter=',')
        for row in zip(sigma_list, mse_train_list, mse_test_list):
            writer_op.writerow(row)
    with open('RBF_methodTwo_random_output.csv', 'w', newline='') as f:
        writer_op1 = csv.writer(f,  delimiter=',')
        for row in zip(sigma_list, mse_train_random_list, mse_test_random_list):
            writer_op1.writerow(row)
    with open('RBF_methodTwo_kmeans_output.csv', 'w', newline='') as f:
        writer_op2 = csv.writer(f,  delimiter=',')
        for row in zip(sigma_list, mse_train_kmeans_list, mse_test_kmeans_list):
            writer_op2.writerow(row)


if __name__ == '__main__':
    main()
