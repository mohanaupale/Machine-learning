"""If the dataset is changed the program needs changes at line 21 to change the number of dataset features in the range function,
and the dataset name i.e. dataset path at line 92"""


import csv
import random
import math
import operator

pred=0;

"""Read the CSV file and generate training and testing datasets"""
def Read_data(file, split, train_vals=[], test_vals=[]):
    with open(file, 'r+') as csvfile:
        data_list = csv.reader(csvfile)
        _original_dataset = list(data_list)
        #print(input_dataset)

#loading data into training and test data sets after converting it to float
        for x in range(len(_original_dataset) - 1):
            for y in range(11): # the value here changes according to the number of features in the dataset 
                _original_dataset[x][y] = float(_original_dataset[x][y])
            if random.random() < split:
                train_vals.append(_original_dataset[x])
            else:
                test_vals.append(_original_dataset[x])


#calculates the euclidian distance for training instance and the testing instance.
def calc_euclideanDistance(inst1, inst2, length):
    distance = 0
    for value in range(length):
        distance += pow((inst1[value] - inst2[value]), 2)
    return math.sqrt(distance)

#identifies the k nearest neighbors
def myknnclassify(X, test, k):
    distances = []
    length = len(test) - 1
    for x in range(len(X)):
        dist = calc_euclideanDistance(test, X[x], length)
        distances.append((X[x], dist))
    distances.sort(key=operator.itemgetter(1))
    nearestneighbors = []
    #print(distances)
    for l in range(k):
        nearestneighbors.append(distances[l][0])
    print(nearestneighbors)

    return nearestneighbors

# calculates the average of the feature of training set(feature to be determined), feature index is stored in the pred.
def myknnregressor(nearestneighbors,pred):
    val1=0
    l=0
    for x in range(len(nearestneighbors)):
        #print(nearestneighbors[x])
        val1=val1+nearestneighbors[x][pred]
    avg=val1/4
    print('Regressor AVG:',avg)

    
#finds the class of the testing instance by finding out which class the max number of neighbors belong to.    
def calc_classVotes(nearestneighbors):
    _Votes = {}
    for x in range(len(nearestneighbors)):
        response = nearestneighbors[x][-1]
        #print(response)
        if response in _Votes:
            _Votes[response] += 1
        else:
            _Votes[response] = 1
    sortedVotes = sorted(_Votes.keys())
    print(sortedVotes)
    return sortedVotes[0]

#calculates the accuracy for myknnclassify function.
def calc_accuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0



def main():
    # prepare data
    train_Set = [] 
    test_Set = []
    split = 0.66                                                # split ratio.generallt 66-34%
    Read_data('glass.csv', split, train_Set, test_Set)          #change the pathname for the CSV file
    print ('Train set: ', repr(len(train_Set)))
    print ('Test set: ', repr(len(test_Set)))
    # generate predictions
    Classify_predictions = []
    k = input("Enter the value for K:")
    k=int(k)
    pred = input("Enter the value of the attribute you want to predict:")    #knn regressor function predicts value of an attribute.
    pred=int(pred)   
    pred=pred-1
    for x in range(len(test_Set)):
        neighbors = myknnclassify(train_Set, test_Set[x], k)
        myknnregressor(neighbors,pred)
        print('Actual:',repr(test_Set[x][pred]))
        result = calc_classVotes(neighbors)
        Classify_predictions.append(result)
        print('> predicted=', repr(result), ', actual=', repr(test_Set[x][-1]))
    accuracy = calc_accuracy(test_Set, Classify_predictions)
    print('Accuracy: ', repr(accuracy) + '%')


main()

#reference:https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
