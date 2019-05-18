
import os
import math
import numpy as np
import cvxopt
import scipy.misc
from sklearn.utils import shuffle
from sklearn import cross_validation


class Soft_SVM():
    #constructor for SVM
    def __init__(self, Slack_val):
        # Defining global variables
        self.slack = Slack_val
        self.lm = None
        self.svm_X = None
        self.svm_y = None
        self.bias = None
        self.weights = None

    def train(self, x, y):
        # Extracting sample and feature lengths
        sample_len, feature_len = x.shape

        # Generating Gramian Matrix
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
        # You can set 'show_progress' to True to see cvxopt output
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
        #print(self.c)
        # Calculating bias
        self.bias = 0.0
        for i in range(len(self.lm)):
            self.bias = self.bias + self.svm_y[i] - np.sum(self.lm * self.svm_y * M[index[i],sv])
        self.bias /= len(self.lm)

        # Calculating weights
        self.weights = np.zeros(feature_len)
        for i in range(len(self.lm)):
            self.weights += self.lm[i] * self.svm_y[i] * self.svm_X[i]

    #prediction for the images.
    def predict(self, x):
        return np.sign(np.dot(x, self.weights) + self.bias)





def Read_images(resize=False):
    X,y = [], []
    #path, direc, docs = next(os.walk('orl_faces'))
    for path, direc, docs in os.walk('att_faces'):
        direc.sort()
        # Iterating through each subject
        for subject in direc:
            files = os.listdir(path+'/'+subject)
            for file in files:
                img = scipy.misc.imread(path+'/'+subject+'/'+file).astype(np.float32)
                if resize:
                    img = scipy.misc.imresize(img, (56, 46)).astype(np.float32)
                X.append(img.reshape(-1))
                y.append(int(subject[1:]))
        X = np.asarray(X)
        #print(X.shape())
        y = np.asarray(y)
    return X, y

#function for cross validating the 
def cross_validate(X, y, c_val):
    X, y = shuffle(X, y)
    #print("len y:",y)
    kf = cross_validation.KFold(len(y), n_folds=5)
    avg_acc_svm = []
    fld = 1
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        print ('\nRunning Fold:', fld , 'for SVM')
        # Defining svm model
        #print(len(x_train))
        #print(len(y_train))
        svm = Soft_SVM(c_val)
        y_train_ovr = [None]*len(y_train)
        y_test_ovr = [None]*len(y_test)
        accuracies = 0
        print ('Running SVM. Please wait...')
        for i in range(1, 41):
        # Setting the selected class as '1' and rest as '-1' depicting the One vs Rest classification.
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
            svm.train(X_train, y_train_ovr)
            predict_class = svm.predict(X_test)
            svm.c = np.sum(predict_class == y_test_ovr)
            accuracies += math.ceil(float(svm.c)/len(predict_class)*100)
            #print("Accuracy test:",accuracies)
        accuracy = accuracies/40
        print ('Accuracy is ', accuracy)
        avg_acc_svm.append(accuracy)
        print("svm:",avg_acc_svm)
        fld += 1
    print ('Average accuracy for SVM ', sum(avg_acc_svm)/5.0, '\n')

        
#for selecting the slack value.
def SVM_Slack_Acceptor():
    print ('\nEnter any number between 1-5(or 0 to quit):\n')
    choice = int(input('1: SVM with c=0.001\n'
                           '2: SVM with c=0.01 \n'
                           '3: SVM with c=1\n'
                           '4: SVM with c=10 \n'
                           '5: SVM with 100 \n'
                           '0: To quit \n'
                            'Please enter your choice: '))
    print ('\n')
    if choice == 1:
        print ("Running SVM with c=0.001\n")
        X, y = Read_images()
        cross_validate(X, y,0.001)
        SVM_Slack_Acceptor()

    elif choice == 2:
        print ("Running SVM with c=0.01\n")
        X, y = Read_images()
        cross_validate(X, y,0.01)
        SVM_Slack_Acceptor()

    elif choice == 3:
        print ("Running SVM with c=1\n")
        X, y = Read_images()
        cross_validate(X, y,1)
        SVM_Slack_Acceptor()

    elif choice == 4:
        print ("Running SVM\ with c=10\n")
        X, y = Read_images()
        cross_validate(X, y,10)
        SVM_Slack_Acceptor()

    elif choice == 5:
        print ("Running SVM with c=100\n")
        X, y = Read_images()
        cross_validate(X, y,100)
        SVM_Slack_Acceptor()

    elif choice == 0:
        quit()

    else:
        print ('Please enter a valid choice')
        SVM_Slack_Acceptor()

def main():
    SVM_Slack_Acceptor()

main()
    
    
    
# References:
# https://www.youtube.com/user/sentdex/videos
# http://cvxopt.org/examples/tutorial/qp.html
# https://pythonprogramming.net/soft-margin-kernel-cvxopt-svm-machine-learning-tutorial/
# http://stackoverflow.com for various syntax doubts.
# https://www.researchgate.net/post/Number_of_folds_for_cross-validation_method    
