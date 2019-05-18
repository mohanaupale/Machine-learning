import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics#representing data in matrix form
%matplotlib inline

np.random.seed(100)
training_data = 1500#training data size
test_data = 500
#separable_features,simulated_labels
def train_data_generator():
    #using random to create the points in multivariate normal distribution.
    distr_1 = np.random.multivariate_normal([1,0],[[1, 0.75],[0.75,1]], training_data)
    distr_2 = np.random.multivariate_normal([0,1.5],[[1, 0.75],[0.75,1]], training_data)
    #now stacking both the arrays vertically together.
    separable_features = np.vstack((distr_1,distr_2)).astype(np.float32)
    labels = np.hstack((np.zeros(training_data), np.ones(training_data)))
    print("\n training data in a scatter plot:\n")
    plt.figure(figsize=(12,8))
    plt.scatter(separable_features[:,0], separable_features[:,1], c = labels, marker='D', alpha = 0.3)
    plt.show()
    return separable_features,labels
    
def test_data_generator():
    #using random to create the points in multivariate normal distribution.
    test_distr_1 = np.random.multivariate_normal([1,0],[[1, 0.75],[0.75,1]], test_data)
    test_distr_2 = np.random.multivariate_normal([0,1.5],[[1, 0.75],[0.75,1]], test_data)
    #now stacking both the arrays vertically together.
    test_separable_features = np.vstack((test_distr_1,test_distr_2)).astype(np.float32)
    test_labels = np.hstack((np.zeros(test_data), np.ones(test_data)))
#print(test_separable_features)
#print(test_simulated_labels)
    print("\n testing data in a scatter plot:\n")
    plt.figure(figsize=(12,8))
    plt.scatter(test_separable_features[:,0], test_separable_features[:,1], c = test_labels, marker='D', alpha = 0.3)
    plt.show()
    return  test_separable_features,test_labels

def sigmoid(scores):
    return 1/(1 + np.exp(-scores))


def cost_function(features, labels, weights):
    observations = len(labels)
    predictions = predict(features, weights)
    #Take the error when label=1
    class1_cost = -labels*np.log(predictions)
    #Take the error when label=0
    class2_cost = (1-labels)*np.log(1-predictions)
    #Take the sum of both costs
    cost = class1_cost - class2_cost
    #Take the average cost
    cost = cost.sum()/observations
    return cost

def predict(features, weights):
    x = np.dot(features, weights)
  #print("dot product:",z)
  #print("sigmoid:",sigmoid(z))
    return sigmoid(x)

def logistic_regression(features, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])
    print("Cost Function:\n")
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 100 == 0:
            print (cost_function(features, target, weights))
           #count=count+1

    return weights


def plot_ROC(test_preds,test_labels):
    fpr, tpr, threshold = metrics.roc_curve(test_labels, test_preds)
    roc_auc = metrics.auc(fpr, tpr)

# method I: plt
#import matplotlib.pyplot as plt
    plt.title('ROC curve (Receiver Operating Characteristic)')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.22f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return

def main():
    separable_features,labels= train_data_generator()
    test_separable_features,test_labels=test_data_generator()
    weights = logistic_regression(separable_features, labels,
                     num_steps = 3000, learning_rate = 0.0001, add_intercept=True)
    #print("count:",count)
#print("\n\n\ntraining weights:",len(weights))
    data_with_intercept = np.hstack((np.ones((separable_features.shape[0], 1)),separable_features))
#print("train data:",data_with_intercept)
    final_scores = np.dot(data_with_intercept, weights)
#print("\n\n\nfinal:",final_scores)
    preds = np.round(sigmoid(final_scores))
#print(preds)
    print ('Training Accuracy: {0}'.format((preds == labels).sum().astype(float) / (0.01 * len(preds))))
    test_data_with_intercept = np.hstack((np.ones((test_separable_features.shape[0], 1)),test_separable_features))
#print("\n\n\n\ndata:",test_data_with_intercept[50])
#print("weights for the testing:",weights)
    test_final_scores = np.dot(test_data_with_intercept, weights)
#print("final test scores:",test_final_scores)
    test_preds = np.round(sigmoid(test_final_scores))
#print(test_preds)
    print ('Testing Accuracy: {0}'.format((test_preds == test_labels).sum().astype(float) / (0.01 * len(test_preds))))
    plot_ROC(test_preds,test_labels)
    
main()    



#reference:https://beckernick.github.io/logistic-regression-from-scratch/