import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as mth

leareningRate = 0.1
#epochs should be 10000 but as that will take a long while, its been replaced with 5
epochs = 5
noofHiddenUnits = 100
momentum = 0.9
examples = 60000

#loads the data from the csv
def data_loader():
    print ("Reading data")
    trainfile = "mnist_train.csv"
    testfile= "mnist_test.csv"
    training_eg = np.array(pd.read_csv(trainfile, header=None), np.float)
    testing_eg = np.array(pd.read_csv(testfile, header=None), np.float)

    training_bias = np.ones((examples, 1), dtype=float);
    testing_bias = np.ones((10000, 1), dtype=float);

    training_eg[:, 1:] = (training_eg[:, 1:] / 255.0)
    testing_eg[:, 1:] = (testing_eg[:, 1:] / 255.0)
    #print("\n\n\ntraining exp:",training_examples)
    training_eg = np.append(training_eg, training_bias, axis=1)
    testing_eg = np.append(testing_eg, testing_bias, axis=1)
    print ("data read and cleaned")
    return training_eg, testing_eg

#calculates the precision based on the confusion matrix
def precision(label, testing_confusion_matrix):
        col = testing_confusion_matrix[:, label]
        return testing_confusion_matrix[label, label] / col.sum()

#calculating recall for each class     
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

#precision for the program based on the data
def precision_macro_average(testing_confusion_matrix):
    rows, columns = testing_confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, testing_confusion_matrix)
    return sum_of_precisions / rows

#Calculating recall for all the classes
def recall_macro_average(testing_confusion_matrix):
    rows, columns = testing_confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, testing_confusion_matrix)
    return sum_of_recalls / columns

        
#accuaracy of the program    
def accuracy(confusion_matrix):
    diagonal_sum= confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum/sum_of_all_elements
    
#calculating true positive based on confusion matrix
    
def T_Positive(testing_confusion_matrix):
    True_Positive = np.diag(testing_confusion_matrix)
    print("true positive:\t")
    print(True_Positive)
    return True_Positive

#calculating false positive based on confusion matrix
def F_Positive(testing_confusion_matrix, True_Positive):
    False_Positive = np.sum(testing_confusion_matrix, axis=0) - True_Positive
    #FP = np.sum(training_confusion_matrix, axis=0)
    print("False Positive:\t")
    print(False_Positive)
    return

#calculating false negative based on confusion matrix
def F_Negative(testing_confusion_matrix):
    False_Negative = np.sum(testing_confusion_matrix, axis=1)
    print("False Negative:\t")
    print(False_Negative)
    return

##calculating true negative, precision recall based on confusion matrix
def T_Negative_precision_recall(testing_confusion_matrix): 
    num_classes = 10
    True_Negative = []
    for i in range(num_classes):
        temp = np.delete(testing_confusion_matrix, i, 0)
        temp = np.delete(temp, i, 1)
        True_Negative.append(sum(sum(temp)))
    print("True Negative:\t")
    print(True_Negative)
    
    print("label precision recall")
    for label in range(10):
        print(f"{label:5f} {precision(label, testing_confusion_matrix):9.3f} {recall(label, testing_confusion_matrix):6.3f}")
    return
    
def main():
    # Hyperparameters...
    training_egs, testing_egs=data_loader() 
       
    outputLayer = np.zeros(noofHiddenUnits + 1)

    outputLayer[0] = 1
    
    training_accuracy = np.zeros(epochs, float)
    testing_accuracy = np.zeros(epochs, float)

    #assigning random weights to the 
    weights_ip_hidden = np.random.uniform(-0.05, 0.05, (785, noofHiddenUnits))
    weights_hidden_out = np.random.uniform(-0.05, 0.05, (noofHiddenUnits + 1, 10))

    old_Delta_H = np.zeros((noofHiddenUnits, 785))
    old_Delta_K = np.zeros((noofHiddenUnits + 1, 10))

    expected_output_vector = np.zeros((examples, 10), float) + 0.1

    
    for i in range(examples):
        expected_output_vector[i][int(training_egs[i][0])] = 0.9
        #print(expected_output_vector)

    hidden_layer_activations = np.zeros(noofHiddenUnits + 1)
    hidden_layer_activations[0] = 1
    
    #training_ANN(training_examples)
       
    for epoch in range(epochs):
        # Reset confusion matrices
        training_confusion_matrix = np.zeros((10, 10), int)
        testing_confusion_matrix = np.zeros((10, 10), int)
        
        print ("Epoch: ", epoch)
        for i in range(examples):
            # Feed forward
            hidden_layer_activations[1:] = (1 / (1 + np.exp(-1 * np.dot(training_egs[i][1:], weights_ip_hidden))))
            outputLayer[1:] = hidden_layer_activations[1:]
            output_layer_activations = (1 / (1 + np.exp(-1 * np.dot(outputLayer, weights_hidden_out))))
            output_layer_error_terms = (output_layer_activations *(1 - output_layer_activations) * (expected_output_vector[i] -output_layer_activations))
            hidden_layer_error_terms = (hidden_layer_activations[1:] * (1 - hidden_layer_activations[1:]) * np.dot(weights_hidden_out[1:, :],output_layer_error_terms))
            deltaK = leareningRate * (np.outer(hidden_layer_activations, output_layer_error_terms)) + (momentum * old_Delta_K)
            deltaH = leareningRate * np.outer(hidden_layer_error_terms, training_egs[i][1:]) + (momentum * old_Delta_H)
            weights_hidden_out = weights_hidden_out + deltaK
            old_Delta_K = deltaK
            weights_ip_hidden = weights_ip_hidden + deltaH.T
            old_Delta_H = deltaH
            training_confusion_matrix[int(training_egs[i][0])][int(np.argmax(output_layer_activations))] += 1
        training_accuracy[epoch] = (float((sum(training_confusion_matrix.diagonal())) / 60000.0) * 100.0)
        print('Epoch ', epoch)
        print('Training Accuracy:', training_accuracy[epoch])
    


        # Claculating for test data
        for i in range(10000):
            # Feed forward pass input to output layer through hidden layer
            hidden_layer_activations[1:] = (1 / (1 + np.exp(-1 * np.dot(testing_egs[i][1:], weights_ip_hidden))))
            # Forward propagate the activations from hidden layer to output layer
            outputLayer[1:] = hidden_layer_activations[1:]
            # calculate dot product for output layer
            # apply sigmoid function to sum of weights times inputs
            output_layer_activations = (1 / (1 + np.exp(-1 * np.dot(outputLayer, weights_hidden_out))))
            #print("output_layer:",output_layer_activations)
            testing_confusion_matrix[int(testing_egs[i][0])][int(np.argmax(output_layer_activations))] += 1

        testing_accuracy[epoch] = ((float(sum(testing_confusion_matrix.diagonal())) / 10000.0) * 100.0)
        print ("Epoch ", epoch, ": ", "Testing Accuracy: ", testing_accuracy[epoch], "%")
    
        #s = np.random.rand(epoch) * 800 + 500
        #plt.scatter(training_accuracy, epoch,s, c="g", alpha=0.5, marker='O',label="Training accuracy")
        

    np.set_printoptions(threshold=np.nan)


    print("Testing confuion matrix")
    print(testing_confusion_matrix)

    T_pos=T_Positive(testing_confusion_matrix)
    F_Positive(testing_confusion_matrix, T_pos)
    F_Negative(testing_confusion_matrix)
    T_Negative_precision_recall(testing_confusion_matrix)
    print("precision total:", precision_macro_average(testing_confusion_matrix))
    print("precision total:", recall_macro_average(testing_confusion_matrix))
    print("accuracy:",accuracy(testing_confusion_matrix))

    

    

    #print("Testing confusion matrix ")
    #print(testing_confusion_matrix)



main()
"""reference:https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
https://github.com/udeodhar/Perceptron"""
