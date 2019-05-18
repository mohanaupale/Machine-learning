# Dependencies
# Please install
# 'CVXOPT' library for python used to solve quadratic equations
# 'sklearn' library used for shuffling the data and cross_validation
# 'scipy' library used for reading images
# 'numpy' library used for various calculations

import os
import numpy as np
import cvxopt
import scipy.misc
import math
from sklearn.utils import shuffle
from sklearn import cross_validation





class Hard_Margin_SVM():

    def __init__(self):
        # Defining global variables
        #Slack= 0 since hard margin svm
        self.slack = 0
        self.lm = None
        self.svm_X = None
        self.svm_y = None
        self.bias = None
        self.weights = None

    def train_svm(self, x, y):
        # Extracting sample and feature lengths
        sample_len, feature_len = x.shape
        #print(feature_len)
        # Generating the Gramian Matrix
        M = np.zeros((sample_len, sample_len))
        for i in range(sample_len):
            for j in range(sample_len):
                M[i,j] = np.dot(x[i], x[j])

        # Calculating values of P, q, A, b, G, h
        P = cvxopt.matrix(np.outer(y, y) * M)
        q = cvxopt.matrix(np.ones(sample_len) * -1)
        A = cvxopt.matrix(y, (1, sample_len), 'd')
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.vstack((np.diag(np.ones(sample_len) * -1), np.identity(sample_len))))
        h = cvxopt.matrix(np.hstack((np.zeros(sample_len), np.ones(sample_len) * self.slack)))

        # Solving quadratic equation
        # if set 'show_progress' to True it shows the solver values for cvxopt output
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        lm = np.ravel(sol['x'])

        # Determining support vectors
        y = np.asarray(y)
        sv = lm > 0.0e-7
        index = np.arange(len(lm))[sv]
        self.lm = lm[sv]
        self.svm_X = x[sv]
        self.svm_y = y[sv]

        # Calculating bias
        self.bias = 0.0
        for i in range(len(self.lm)):
            self.bias = self.bias + self.svm_y[i] - np.sum(self.lm * self.svm_y * M[index[i],sv])
        self.bias /= len(self.lm)

        # weight calculator
        self.weights = np.zeros(feature_len)
        for i in range(len(self.lm)):
            self.weights += self.lm[i] * self.svm_y[i] * self.svm_X[i]

    def predict(self, x):
        return np.sign(np.dot(x, self.weights) + self.bias)



def Read_Images(resize=False):
        X,y = [], []
        """Reading the data in the form of files from the directory"""
        for path, directory, docs in os.walk('att_faces'):
            directory.sort()
            # Iterations for each subject
            for subject in directory:
                files = os.listdir(path+'/'+subject)
                for file in files:
                    img = scipy.misc.imread(path+'/'+subject+'/'+file).astype(np.float32)
                    if resize:
                        img = scipy.misc.imresize(img, (56, 46)).astype(np.float32)
                    X.append(img.reshape(-1))
                    y.append(int(subject[1:]))
            X = np.asarray(X)
            y = np.asarray(y)
        return X, y

   
    
def cross_val(x, y):
    #suffling x and y.
    x, y = shuffle(x, y)
    #print("len y:",y)
    kf = cross_validation.KFold(len(y), n_folds=5)
    avg_acc_svm = []
    fld = 1
    for train, test in kf:
        X_train, X_test, y_train, y_test = x[train], x[test], y[train], y[test]
        print ('\nRunning Fold:', fld , 'for the Hard Margin SVM:')
        # Defining svm model
        svm = Hard_Margin_SVM()
        y_train_ovr = [None]*len(y_train)
        y_test_ovr = [None]*len(y_test)
        accuracies = 0
        print ('The Hard-Margin SVM. Please wait...')
        for i in range(1, 41):
            # Setting the selected class as '1' and rest as '-1' for the One vs Rest classification.
            for j in range(0, 320):
                if y_train[j] == (i):
                    y_train_ovr[j] = 1
                else:
                    y_train_ovr[j] = -1
            for j in range(0, 80):
                if y_test[j] == (i):
                    y_test_ovr[j] = 1
                else:
                    y_test_ovr[j] = -1

                    # Taking Set_A as training set and Set_B for testing
            svm.train_svm(X_train, y_train_ovr)
            predict_class = svm.predict(X_test)
            svm.c = math.ceil(np.sum(predict_class == y_test_ovr))
                #print(c)
            accuracies += math.ceil(float(svm.c)/len(predict_class)*100)
                    #print("Accuracy test:",accuracies)
        accuracy = accuracies/40
        print ('Accuracy is ', accuracy)
        avg_acc_svm.append(accuracy)
        print("svm:",avg_acc_svm)
        fld += 1
    print ('Average accuracy for SVM ', sum(avg_acc_svm)/5.0, '\n')

       
"""Main function to read the images and run the Hard_margin svm code"""        
def main():
       
    #print (" SVM\n")
    x, y = Read_Images()
    cross_val(x, y)
    #task_selector()
 
    

   

        
main()





# References:
# https://www.youtube.com/user/sentdex/videos
# http://cvxopt.org/examples/tutorial/qp.html
# https://pythonprogramming.net/soft-margin-kernel-cvxopt-svm-machine-learning-tutorial/
# http://stackoverflow.com for various syntax doubts.
# https://www.researchgate.net/post/Number_of_folds_for_cross-validation_method
