from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import warnings
import matplotlib as mlt

import matplotlib.pyplot as plt
#warnings.filterwarnings("ignore", category=DeprecationWarning)
mx.random.seed(200)
num_inputs = 784
num_outputs = 10
num_examples = 60000

#creating 2 contexts for training and testing.
train_ctx = mx.cpu()
test_ctx = mx.cpu()

weights = nd.random_normal(shape=(num_inputs, num_outputs),ctx=test_ctx)
bias = nd.random_normal(shape=num_outputs,ctx=test_ctx)

True_Positive = np.zeros(10, int)
False_Positive= np.zeros(10, int)
False_Negative = np.zeros(10, int)

#transforming the data for convinience of working
def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

#obtains the weights/probabilitis from all the the branches and feeds it to softmax function.
def net(X):
    y_linear = nd.dot(X, weights) + bias
    yhat = softmax(y_linear)
    return yhat

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
    return

#impleents the softmax function to determine if a data point will belong to a perticular class.
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear, axis=1).reshape((-1,1)))
    norms = nd.sum(exp, axis=1).reshape((-1,1))
    return exp / norms

# a cost function calculating the error
def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat+1e-6))
#Calculating accuracy of the 
def calc_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(test_ctx).reshape((-1,784))
        label = label.as_in_context(test_ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements

def model_predict(net,data):
    output = net(data)
    return nd.argmax(output, axis=1)

def calc_TP(testing_confusion_matrix):    
# Calculating True Positives
    for i in range(10):
        for j in range(10):
            if (i == j):
                True_Positive[i] += testing_confusion_matrix[i,j]
    print("True_Positive:",True_Positive)
# Calculating False Positives

def calc_FP(testing_confusion_matrix):
    for i in range(10):
        False_Positive[i] = sum(testing_confusion_matrix[:,i])-testing_confusion_matrix[i,i]
    print("False_Positive:",False_Positive)
    return

def calc_FN(testing_confusion_matrix):    
# Calculating False Negatives
    for i in range(10):
        False_Negative[i] = sum(testing_confusion_matrix[i, :], 2)-testing_confusion_matrix[i,i]
    print("False_Negative:",False_Negative)
    return
#Caluclating Recall
def calc_recall():
    recall = np.zeros(10,float)
    for i in range(10):
        recall[i] = float(True_Positive[i])/(float(True_Positive[i]) + float(False_Negative[i]))

    print(recall)
    print("Final Recall:")
    re = 0.0
    for i in range(10):
        re += recall[i]
    print(re/10.0)
    return

#Calculating Precision
def calc_prec():
    prec = np.zeros(10, float)
    for i in range(10):
        prec[i] = float(True_Positive[i])/(float(True_Positive[i]) + float(False_Positive[i]))

    print(prec)
    print("Final Precision:")
    pf = 0.0
    for i in range(10):
        pf += prec[i]
    print(pf/10.0)
    return









def main():
    mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
    #print(mnist_train)
    mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
    batch_size = 64
    train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    #print(train_data)
    test_data = mx.gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)


   
    params = [weights, bias]

    for param in params:
        param.attach_grad()

    sample_y_linear = nd.random_normal(shape=(2,10))
    sample_yhat = softmax(sample_y_linear)
    #print(sample_yhat)

    #print(nd.sum(sample_yhat, axis=1))
    calc_accuracy(test_data, net)
#epochs should be 10000 but as that will take a long while, its been replaced with 20
    epochs = 20
    learning_rate = .005
#loop that learns the training data and predicts the probability of each image to belong in each of th 1-9 classes
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(test_ctx).reshape((-1,784))
            label = label.as_in_context(test_ctx)
            label_one_hot = nd.one_hot(label, 10)
            with autograd.record():
                output = net(data)
                loss = cross_entropy(output, label_one_hot)
            loss.backward()
            SGD(params, learning_rate)
            cumulative_loss += nd.sum(loss).asscalar()

        test_accuracy = calc_accuracy(test_data, net)
        train_accuracy = calc_accuracy(train_data, net)
        print("Epoch %s. \tLoss: %s \tTrain_acc %s \tTest_acc %s" % (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))

# Define the function to do prediction

# let's sample 10 random data points from the test set
    sample_data = mx.gluon.data.DataLoader(mnist_test, 10000 , shuffle= False)
    pred = np.zeros(10000, int)
    for i, (data, label) in enumerate(sample_data):
        data = data.as_in_context(test_ctx)
        #print("data.shape")
        #print(data.shape)
        #im = nd.transpose(data,(1,0,2,3))
        #im = nd.reshape(im,(28,10*28,1))
        #imtiles = nd.tile(im, (1,1,3))

        #plt.imshow(imtiles.asnumpy())
        #plt.show()
        pred=model_predict(net,data.reshape((-1,784)))
        #print('model predictions are:', pred)
    predict = pred.asnumpy()
    testing_confusion_matrix = np.zeros((10, 10), int)
    for i in range(10000):
        image, label = mnist_test[i]
        label1=int(label)
        y = int(predict[i])
        #y = int(pred[i])
        testing_confusion_matrix[label1, y] += 1
    print("testing_confusion_matrix:")
    print(testing_confusion_matrix)
    print("sanity check:",sum(map(sum, testing_confusion_matrix)))
    calc_TP(testing_confusion_matrix)
    calc_FP(testing_confusion_matrix)
    calc_FN(testing_confusion_matrix)
    calc_recall()
    calc_prec()
    print("accuracy:", accuracy(testing_confusion_matrix))
main()

#reference: http://gluon.mxnet.io/chapter02_supervised-learning/softmax-regression-scratch.html