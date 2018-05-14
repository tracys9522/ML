'''
    File: regression.py
    Desciption: Perform linear regression and ridge regression analysis on training and test data set
    Author: Tracy Sun
'''

import math
import numpy as np
import re

#number of test & training data
nTraining = 1595
nTest = 399
nXi = 95    #num of features

# read txt file into data
def read_file(file_name):
    data = []
    # read every float number into data sets
    with open(file_name,'r') as f:
        for line in f:
            line = re.findall(r'[-+]?\d+\.\d+', line)
            if not line:
                continue
            else:
                data.append(line)
        f.close()

    data = [[float(float(num)) for num in row] for row in data]
    data = np.array(data)
    return data

# calculate w
def calc_w(x,x_tran,y,lamda):
    iden = np.identity(nXi+1)
    a = np.linalg.inv(np.dot(x_tran,x)+(lamda*iden))
    b = np.dot(x_tran,y)
    w = np.dot(a,b)
    return w

def calc_rmse(y1,y2,m):
    return math.sqrt(sum((y1-y2)**2)/m)

# gradient decent
def gradient_decent(w_t,l_rate,x,x_trans,y,ep,lam):
    
    while 1:
        w_t1 = w_t + l_rate * (np.dot(x_trans,(y - np.dot(x,w_t))) - lam * w_t)
        sum_error = 0
        for i in range(96):
            sum_error += float(abs(w_t[i] - w_t1[i]))
        sum_error /= 96
        if (sum_error <= ep):
            break
        w_t = w_t1
    return w_t


if __name__ == '__main__':
    
    train_file = "crime-train.txt"
    test_file = "crime-test.txt"

    # training and test data set
    train_data = read_file(train_file)
    test_data = read_file(test_file)
    a = np.ones((nTraining,1))
    b = np.ones((nTest,1))
    train_data = np.column_stack((train_data, a))
    test_data = np.column_stack((test_data, b))

    y_train = train_data[:,0]
    y_train = np.reshape(y_train,(nTraining,1))
    
    y_test = test_data[:,0]
    y_test = np.reshape(y_test,(nTest,1))
   
    # insert column of 1s at the last column of X
    x_train = train_data[:,1:]
    x_test = test_data[:,1:]
    
    x_traintrans = x_train.transpose()
    x_testrans = x_test.transpose()
    
    # perform linear regression close form and calculate w
    w = calc_w(x_train,x_traintrans,y_train,0)
    
    #calculate Yi
    yi_train = np.dot(x_train,w)
    yi_test = np.dot(x_test,w)
    
    # RMSE for both training and test data
    train_rmse = calc_rmse(yi_train,y_train,nTraining)
    test_rmse = calc_rmse(yi_test,y_test,nTest)
    
    # ridge regression using k fold cross validation
    lam = 400
    best_lambda = lam
    k = 5
    n = nTraining/5    #k fold data size
    min_rmse = 1
    
    x_ridge = np.reshape(x_train,(5,n,nXi+1))
    y_ridge = np.reshape(y_train,(5,n,1))
    
    # ridge regression test data sets
    x1 = x_ridge[1:]
    x2 = x_ridge[[0,2,3,4]]
    x3 = x_ridge[[0,1,3,4]]
    x4 = x_ridge[[0,1,2,4]]
    x5 = x_ridge[:-1]
    
    y1 = y_ridge[1:]
    y2 = y_ridge[[0,2,3,4]]
    y3 = y_ridge[[0,1,3,4]]
    y4 = y_ridge[[0,1,2,4]]
    y5 = y_ridge[:-1]
    
    # cut lambda value by factor of 2 and find optimal lambda with minimum rmse rate
    for attemp in range(10):
        
        for k in range(5):
            ridge_rmse = 0
            
            validation = x_ridge[k]
            validation = np.reshape(validation,(n,nXi+1))   #validation set
            
            y_validation = y_ridge[k]
            y_validation = np.reshape(y_validation,(n,1))   #corresponing y
            
            if k == 0:
                x = x1
                y = y1
            elif k==1:
                x = x2
                y = y2
            elif k==2:
                x = x3
                y = y3
            elif k==3:
                x = x4
                y = y4
            else:
                x = x5
                y = y5
            
            y = np.reshape(y,(nTraining-n,1))
            x = np.reshape(x,(nTraining-n,nXi+1))
            
            y_train = np.reshape(y_train,(nTraining,1))
            xtrans = x.transpose()
            
            w_ridge = calc_w(x,xtrans,y,lam)
            yi_ridge = np.dot(validation,w_ridge)
            
            # sum of all rmse for each lambda value
            ridge_rmse += calc_rmse(yi_ridge,y_validation,n)
    

        ridge_rmse/=5
        print "lambda = ",lam
        print "RMSE = ",ridge_rmse
        #find the smallest rmse that leads to the best lambda value
        if ridge_rmse < min_rmse:
            min_rmse = ridge_rmse
            best_lambda = lam
        
        lam = float(lam/2)

    # train the entire training data with the most optimal lambda
    w_ridge = calc_w(x_train,x_traintrans,y_train,best_lambda)
    yi_testridge = np.dot(x_test,w_ridge)
    ridge_testrmse = calc_rmse(yi_testridge,y_test,nTest)
    
    # gradient decent algorithm for linear regression
    l_rate = 0.00001     #learning rate
    ep = math.pow(10,-6) #converge criteria

    w_t = np.zeros((96,1))
    w_t = gradient_decent(w_t,l_rate,x_train,x_traintrans,y_train,ep,0)
    #print np.linalg.norm(w_t - w)   #norm difference of w_t vs w

    #compute RMSE for training and test data
    lrgd_trainyi = np.dot(x_train,w_t)
    lrgd_testyi = np.dot(x_test,w_t)
    lrgd_trainrmse = calc_rmse(lrgd_trainyi,y_train,nTraining)
    lrgd_testrmse = calc_rmse(lrgd_testyi,y_test,nTest)
    
    # compute RMSE for test data on ridge regression
    ridge_wt = np.zeros((96,1))
    ridge_wt = gradient_decent(ridge_wt,l_rate,x_test,x_testrans,y_test,ep,best_lambda)
    rrgd_testyi = np.dot(x_test,ridge_wt)
    rrgd_testrmse = calc_rmse(rrgd_testyi,y_test,nTest)
    
    #print all important info for LR and RR analysis
    print("\nx_training data\n*************************************")
    print np.shape(x_train)
    print x_train
    
    print("\nx_test data\n*************************************")
    print x_test
    
    print("\ncalculated w\n*************************************")
    print np.shape(w)
    print w
    
    print("\nLinear Regression RMSE\n*************************************")
    print "Linear Regression RMSE train = ",train_rmse
    print "Linear Regression RMSE test = ",test_rmse
    
    print("\nK Fold Analysis on Ridge regression\n*************************************")
    print "most optimal lambda = ",best_lambda
    print "Ridge RMSE for test data: ",ridge_testrmse
    
    print("\nGradient Decent linear Regression\n*************************************")
    print "norm value of w_t vs w: ",np.linalg.norm(w_t - w)
    print "linear regression gradient decent RMSE Train: ",lrgd_trainrmse
    print "linear regression gradient decent RMSE Test: ",lrgd_testrmse
    
    print("\nGradient Decent Ridge Regression\n*************************************")
    print "norm value of w_t vs w: ",np.linalg.norm(ridge_wt - w)
    print "ridge regression gradient decent RMSE Test: ",rrgd_testrmse
