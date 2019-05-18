from math import log
#import matplotlib as plt
import numpy as np


_dataset = 'mushroom.csv'
_attributes = 'agaricus-lepiota.names'

attributes_edible_list = []
attributes_poisonous_list = []

edible_dataset = []
poisonous_dataset = []

poison=0
edible=0

training_data = []
test_data = []

g_attributes = [] #doesn't include poisonous or edible columns
g_attributes_dictionary = {}


#loads the datasets and divides it into test and train data. revmoves the "?" instances. 
def load_datasets():
    counter=0
    with open(_dataset, 'r+') as dataset_file:
        dataset_lines = dataset_file.readlines()
        
    for line in dataset_lines:
        attributes = line.split(',')
        #print(attributes)
        # Get rid of newline character on last attribute
        attributes[-1] = attributes[-1].strip()
        #if(attributes[11]=='?'):
         #   attributes.pop()
          #  print(len(attributes))
        
        if(attributes[11]=='?'):
                #attributes.pop()
                continue
                #print("attribute popped")
        else:
                if attributes[0] == 'e':
                    edible_dataset.append((attributes[0], attributes[1:]))
                    #print (positive_dataset)
                    if(counter<4000):
                        training_data.append(edible_dataset.pop())
                        counter+=1
                    else:
                        test_data.append(edible_dataset.pop())
            
                else:
                    poisonous_dataset.append((attributes[0], attributes[1:]))
                    if(counter<4000):
                        training_data.append(poisonous_dataset.pop())
                        counter+=1
                    else:
                        test_data.append(poisonous_dataset.pop())
    print("training dataset size:",len(training_data))
    print ("testing dataset size:",len(test_data))
    print("total data:",len(training_data+test_data))
    
# retrieves all the values a feature can take from the names file              
def Features():
    with open(_attributes, 'r+') as attributes_file:
        attributes_lines = attributes_file.readlines()
    for line in attributes_lines:
        pair = line.strip().split()
        g_attributes.append(pair[0])
        g_attributes_dictionary[pair[0]] = pair[1].split(',')
   # print("\n\n\n\t\tg_attributes",g_attributes)
   # print ("\n\n\n\tg_dictionary",g_attributes_dictionary)

#the features of all the instances are sorted based on edible and poisonous and stored in to list accodingly.
def Feature_Lists():
    attr_count = 0
    val_count = 0
    
    for i in range(len(g_attributes)):
        attributes_edible_list.append([])
        attributes_poisonous_list.append([])
        
    for i in attributes_edible_list:
        for j in range(21):
            i.append(0)
            #print("\n\n\n\n\n\n\n\n\ni value", i)
    for i in attributes_poisonous_list:
        for j in range(21):
            i.append(0)
    
    for attr in g_attributes:
        val_count = 0
        for value in g_attributes_dictionary[attr]:
            for example in training_data:
                if value == example[1][attr_count] and example[0] == 'e':
                    attributes_edible_list[attr_count][val_count] += 1
            val_count += 1
        attr_count += 1
    attr_count = 0
    
    for attr in g_attributes:
        val_count = 0
        for value in g_attributes_dictionary[attr]:
            for example in training_data:
                if value == example[1][attr_count] and example[0] == 'p':
                    attributes_poisonous_list[attr_count][val_count] += 1
            val_count += 1
        attr_count += 1

#this is the naive bayes function for calculating the probability for each feature and the entire probability.
def naive_bayes(instance,neg,pos):
    count = 0
    edible_prob = 1.0
    poisonous_prob = 1.0
    
    for attr in instance:
        edible_prob *= attributes_edible_list[count][g_attributes_dictionary[g_attributes[count]].index(attr)]
        poisonous_prob *= attributes_poisonous_list[count][g_attributes_dictionary[g_attributes[count]].index(attr)]
           
        count += 1
    if poisonous_prob > edible_prob: 
        return 'p'
    else:
        return 'e'

    
if __name__ == '__main__': 
    
    load_datasets()
    Features()
    Feature_Lists()
    print ("Tables uplaoded successfully!!!")
    
    num_edible = 0
    num_poisonous = 0
    
    
    for i in training_data: 
        if i[0] == 'e':
            num_edible += 1
            pos_train.append(i[1])
        else:
            num_poisonous += 1
            neg_train.append(i[1])
    print ("total edible in training data:",num_edible)
    print ("total poisonous in training data:",num_poisonous)
        
    correct = 0    #for calculation of accuracy
    
    for instances in test_data:
        actual = instances[0]
        calculated = naive_bayes(instances[1],num_poisonous,num_edible)
        print ("actual: %s classified: %s "%(actual, calculated))
        if actual=="p":
            poison+=1
        else:
            edible+=1
        if actual == calculated:
            correct += 1
            
    print ("total edible:",edible)
    print ("total poison:",poison)
    print ("Percent correct: %f "%(float(correct*100)/float(len(test_data))))
    
#reference:https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/