#!/usr/bin/env python
# coding: utf-8

# In[9]:


from cvxpy import *
from numpy import array
import numpy
import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
from mlxtend.data import loadlocal_mnist
import time
import gzip


# In[7]:


def plotting(x,y,a,b,t):
    #  Hyperplane a'x -b = 0, t is the width
    
    plt.ylim([-5,5])
    plt.xlim([-5,5])
    plt.scatter(x[:,0], x[:,1], c='b', marker='o')
    plt.scatter(y[:,0], y[:,1], c='r', marker='x')
    d1_min = np.min([x[:,0]])
    d1_min1 = np.min([y[:,0]])
    if d1_min > d1_min1:
        d1_min=d1_min1
    d1_max = np.max([x[:,0]])
    d1_max1 = np.max([y[:,0]])
    if d1_max < d1_max1:
        d1_max=d1_max1
    # Line form: (-a[0] * x - b ) / a[1]
    d2_atD1min = (-a[0]*d1_min + b ) / a[1]
    d2_atD1max = (-a[0]*d1_max + b ) / a[1]

    sup_up_atD1min = (-a[0]*d1_min + b + t ) / a[1]
    sup_up_atD1max = (-a[0]*d1_max + b + t ) / a[1]
    sup_dn_atD1min = (-a[0]*d1_min + b - t ) / a[1]
    sup_dn_atD1max = (-a[0]*d1_max + b - t ) / a[1]

    # Plot the Lines
    plt.plot([d1_min,d1_max],[d2_atD1min,d2_atD1max],color='black')
    plt.plot([d1_min,d1_max],[sup_up_atD1min,sup_up_atD1max],'--',color='gray')
    plt.plot([d1_min,d1_max],[sup_dn_atD1min,sup_dn_atD1max],'--',color='gray')
##########################
def createdata(x):
    if x > 0:
        np.random.seed(71169)
        #np.random.seed(20610093471169)
        xp = np.random.multivariate_normal([1,1],[[1,0], [0,1]],200)
        xn = np.random.multivariate_normal([-1,-1],[[1,0], [0,1]],200)
        yp = np.ones(200)
        yn = np.multiply(np.ones(200),-1)
    else:
        np.random.seed(71169)
        #np.random.seed(20610093471169)
        xp = np.random.multivariate_normal([1,1],[[1,0], [0,1]],200)
        xn = np.random.multivariate_normal([-1,-1],[[1,0], [0,1]],200)
        yp = np.ones(200)
        yn = np.zeros(200)
    return xp,yp,xn,yn


# In[8]:


##########
#Optimization
##########
def opt(c,xp,xn,yp,yn,m,n,p,pw,cross,tx,ty,ret):
    #preprocess
    w = Variable(d)
    b = Variable()
    zipp = Variable(m)
    zin = Variable(n)
    C = Parameter()
    C.value = c
    xp=np.array(xp)
    yp=np.array(yp)
    xn=np.array(xn)
    yn=np.array(yn)
    
    #Minimizing norm^2/2 + C*zi
    obj = Minimize((norm(w,2)**2/2) + C*np.ones(m)*zipp + C*np.ones(n)*zin)
#####
#####Print out the constraints value
    constraints = [w.T * xp[i] + b >= 1 - zipp[i] for i in range(m)]
    constraints.extend([w.T * xn[i] + b <= -1 + zin[i] for i in range(n)])
    constraints.extend([zipp[i] >= 0  for i in range(m)])
    constraints.extend([zin[i] >= 0  for i in range(n)])
    #print("Solving...")
    prob = Problem(obj, constraints)
    prob.solve(qcp=True)
    w = w.value
    normw=np.linalg.norm(w)
    #thi=np.arctan(w[1]/w[0])
    b = b.value
    c = C.value
    #print("Solved!")
    error = 0
    #print("wx + b >=0:")
    # if dual_value = 0, it's not a support vector
    for i in range(m):
        x = np.dot(w.T,xp[i])+b
        #print(x)
        if x<0:
            error+=1
    #print("wx + b < 0:")
    for i in range(n):
        x = np.dot(w.T,xn[i])+b
        #print(x)
        if x>=0:
            error+=1
    #print C and vector w and dual option 
    if pw>0:
        print("C=", c)
        print("w vector=",w)
        print("margin is 1/||w||=",1/normw)
        i=0
        #plotting option
    if p>0:
        plotting(xp,xn,w,b,1)
        plt.show()
        #print(constraints[399].dual_value)
        for i in range(400):
            if (constraints[i].dual_value !=0) and i<200 and (np.dot(w.T , xp[i]) + b > 1 - zipp[i].value) :
                print(xp[i],"is support vector")
            elif (constraints[i].dual_value !=0) and (np.dot(w.T , xn[i-200]) + b < (zin[i-200].value-1)) and i>=200:
                print(xn[i-200],"is support vector")
        # cross validation option
    if cross>0:
        x = np.dot(w.T,tx)+b
        #print(x)
        if x>0:
            if ty<0:
                #print("wx +b=",x,",but y is",ty)
                return 1
            else:
                return 0
        if x<0:
            if ty>0:
                #print("wx +b=",x,",but y is",ty)
                return 1
            else:
                return 0
    else:
        print("SVM empirical Error:",error,"/",m+n,"= {:.3f} %\n".format(error/(m+n)*100))
    if ret == 0:
        return error
    else:
        return w,b
###########
def crossvalid(xp,yp,m,n,p,pw):
    error=0
    #leave one out
    for i in range(m):
        xpp=list(xp)
        ypp=list(yp)
        testdatax=xpp[i]
        testdatay=ypp[i]
        del xpp[i]
        del ypp[i]
        error = error + opt(1,xpp,xn,ypp,yn,m-1,n,p,pw,1,testdatax,testdatay,0)
        #print("error=",error)
    for i in range(n):
        xnn=list(xn)
        ynn=list(yn)
        testdatax=xnn[i]
        testdatay=ynn[i]
        del xnn[i]
        del ynn[i]
        error = error + opt(1,xp,xnn,yp,ynn,m,n-1,p,pw,1,testdatax,testdatay,0)
        #print("error=",error)
    print("Cross Validation Error: {:.3f} %".format(error/(m+n)*100))


# In[5]:


d = 2 #dimension
xp=[]
xn=[]
yp=[]
yn=[]

xp,yp,xn,yn=createdata(1)
opt(0.01,xp,xn,yp,yn,200,200,1,1,0,0,0,0)
opt(100000,xp,xn,yp,yn,200,200,1,1,0,0,0,0)


# In[5]:


d = 2 #dimension
xp=[]
xn=[]
yp=[]
yn=[]
m=200
n=200
start = time.time()
xp,yp,xn,yn=createdata(1)
crossvalid(xp,yp,m,n,0,0)
end = time.time()
print("Time elapsed while crossvalid: {:.3f} ".format(end - start),"s")


# In[44]:



d = 784
# read data and preprocess
from mnist.loader import MNIST

mndata = MNIST('')

X_train , Y_train = mndata.load_training()
# or
X_test, Y_test = mndata.load_testing()
xp=np.empty((0,784), int)
yp=np.empty((0,1),int)
xn=np.empty((0,784), int)
yn=np.empty((0,1),int)
X_opt=np.empty((0,784), int)
Y_opt=np.empty((0,1),int)
for i in reversed(range(len(Y_train))):
    if Y_train[i] ==1:
        xp=np.append(xp,[X_train[i]],axis=0)
        yp=np.append(yp,[Y_train[i]])
    elif Y_train[i] == 0:
        xn=np.append(xn,[X_train[i]],axis=0)
        yn=np.append(yn,[Y_train[i]])

for i in reversed(range(len(Y_test))):
    if Y_test[i] ==1 or Y_test[i] == 0:
        X_opt=np.append(X_opt,[X_test[i]],axis=0)
        Y_opt=np.append(Y_opt,[Y_test[i]])
print(len(xp)+len(xn))
print("start to optimize")
start = time.time()
error = 0
w, b = opt(1,xp,xn,yp,yn,len(xp),len(xn),0,0,0,0,0,1)
print("optimization finished")
for i in range(len(Y_opt)):
    x = np.dot(w.T, X_opt[i])+b
    if x>0:
        if Y_opt[i]==0:
            #print("wx +b=",x,",but y is",ty)
            error = error +1
        else:
            error = error
    if x<0:
        if Y_opt[i]>0:
            error = error +1
        else:
            error = error
print("MNIST error {:.3f} %".format(error/len(Y_opt)*100))
end = time.time()
print("Time elapsed: {:.3f} ".format(end - start),"s")


# In[10]:


def plotlinear(x,y,a,b,t):
    #  Hyperplane a'x -b = 0, t is the width
    
    plt.ylim([-5,5])
    plt.xlim([-5,5])
    plt.scatter(x[:,0], x[:,1], c='b', marker='o')
    plt.scatter(y[:,0], y[:,1], c='r', marker='x')
    d1_min = np.min([x[:,0]])
    d1_min1 = np.min([y[:,0]])
    if d1_min > d1_min1:
        d1_min=d1_min1
    d1_max = np.max([x[:,0]])
    d1_max1 = np.max([y[:,0]])
    if d1_max < d1_max1:
        d1_max=d1_max1
    # Line form: (-a[0] * x - b ) / a[1]
    d2_atD1min = (-a[0]*d1_min + b ) / a[1]
    d2_atD1max = (-a[0]*d1_max + b ) / a[1]
    # Plot the Lines
    plt.plot([d1_min,d1_max],[d2_atD1min,d2_atD1max],color='black')
####################################
def linear_regression(xp,yp,xn,yn,p):
    x = np.concatenate((xp,xn))
    one = np.ones(x.shape[0])
    one = one.reshape(x.shape[0],1)
    
    x = (np.concatenate((one.T, x.T)))
    x = x.T
    yp = np.array(yp)
    yn = np.array(yn)
    
    y = np.concatenate((yp,yn))
    w = np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y)
    x = np.concatenate((xp,xn))
    b = w[0]
    w = w[-(w.shape[0]-1):]
    
    # sign(w * x) - y (set 0 as threshold)
    error = np.sum(np.abs(np.sign(np.dot(w.reshape(1,x.shape[1]),x.T))- y)/2)
    # if value is not equal, the value after subtraction would be doubled
    if p > 0:
        print("w=",w,"b=",b)
        plotlinear(xp,xn,w,b,1)
        print("Linear regression empirical error=",error,"/",x.shape[0],"={:.3f} %".format(error/x.shape[0]*100))
    return w
##########################
def linear_regressionnmist(x,y):
    w = np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y)
    b = w[0]
    w = w[-(len(x)-1):]
    return w,b


# In[32]:


d = 784
# read data and preprocess
from mnist.loader import MNIST

mndata = MNIST('')

X_train , Y_train = mndata.load_training()
# or
X_test, Y_test = mndata.load_testing()
xp=np.empty((0,785), int)
yp=np.empty((0,1),int)
X_opt=np.empty((0,784), int)
Y_opt=np.empty((0,1),int)
for i in reversed(range(len(Y_train))):
    if Y_train[i] ==1 or Y_train[i] == 0:
        temp = np.sum([X_train[i],0.00001*np.random.rand(1, 784)],axis=0)
        temp = np.append([1],[temp])
        xp=np.append(xp,[temp],axis=0)
        yp=np.append(yp,[Y_train[i]])
for i in reversed(range(len(Y_test))):
    if Y_test[i] ==1 or Y_test[i] == 0:
        X_opt=np.append(X_opt,[X_test[i]],axis=0)
        Y_opt=np.append(Y_opt,[Y_test[i]])
print("start to optimize")
start = time.time()
error = 0
w,b= linear_regressionnmist(xp,yp)
w = w[-(len(w)-1):]
print("optimization finished")
for i in range(len(Y_opt)):
    x = np.dot(w.T, X_opt[i])+b
    if x>0:
        if Y_opt[i]==0:
            #print("wx +b=",x,",but y is",ty)
            error = error +1
        else:
            error = error
    if x<0:
        if Y_opt[i]>0:
            error = error +1
        else:
            error = error
print("MNIST error {:.3f} %".format(error/len(Y_opt)*100))
end = time.time()
print("Time elapsed: {:.3f} ".format(end - start),"s")


# In[15]:


xp,yp,xn,yn=createdata(1)
w = linear_regression(xp,yp,xn,yn,1)


# In[7]:


xp,yp,xn,yn=createdata(1)
error = 0
for i in range(xp.shape[0]):
    xpp=np.matrix(xp)
    ypp=np.matrix(yp)
    testdatax = xpp[i,:]
    testdatay = ypp[:,i]
    xpp = np.delete(xpp,i,0)
    ypp = np.delete(ypp,i)
    w = linear_regression(xpp,ypp.T,xn,yn.reshape(len(yn),1),0)
    error += int(np.sum(np.abs(np.sign(np.dot(w.reshape(1,2),testdatax.T)) - testdatay)/2)) 
for i in range(xn.shape[0]):
    xnn=np.matrix(xn)
    ynn=np.matrix(yn)
    testdatax = xnn[i,:]
    testdatay = ynn[:,i]
    xnn = np.delete(xnn,i,0)
    ynn = np.delete(ynn,i)
    w = linear_regression(xp,yp.reshape(len(yp),1),xnn,ynn.T,0)
    error += int(np.sum(np.abs(np.sign(np.dot(w.reshape(1,2),testdatax.T)) - testdatay)/2))
print("Linear regression crossvalidation error:", error,"/",xp.shape[0]+xn.shape[0],"= {:.3f} %".format(error/(xp.shape[0]+xn.shape[0])*100))


# In[23]:


#Logistic_regression
#######################################
def errorf(x, y):
    if x > 0:
        if y ==1:
            return 0
        else:
            return 1
    if x <= 0:
        if y ==1:
            return 1
        else:
            return 0
##########################
def logistic_regression(xp,yp,xn,yn,d,p):
    x = np.concatenate((xp,xn))
    y = np.concatenate((yp,yn))
    y = y.tolist()
    beta = cp.Variable(d)
    log_likelihood = cp.sum(
        cp.multiply(y, x @ beta) - cp.logistic(x @ beta)
    )
    problem = cp.Problem(cp.Maximize(log_likelihood/d))
    problem.solve()
    #beta = beta.value
    errort = 0
    for i in range(len(y)):
        
        errort = errort + errorf( (x[i] @ beta).value, y[i])
    # if value is not equal, the value after subtraction would be doubled
    if p > 0:
        print("beta=",beta.value)
        plotlinear(xp,xn,beta.value,0,1)
        print("Logistic regression empirical error=","={:.3f} %".format(errort/x.shape[0]*100))
    return beta.value
#############################
def logistic_nmistregression(x,y,d):
    beta = cp.Variable(d)
    log_likelihood = cp.sum(
        cp.multiply(y, x @ beta) - cp.logistic(x @ beta)
    )
    problem = cp.Problem(cp.Maximize(log_likelihood/d))
    problem.solve(solver = SCS,verbose=True)
    #beta = beta.value
    errort = 0
    for i in range(len(y)):
        errort = errort + errorf( (x[i] @ beta).value, y[i])
    print("Logistic regression empirical error=","={:.3f} %".format(errort/x.shape[0]*100))
    return beta.value
##########################################


# In[25]:


d = 784
# read data and preprocess
from mnist.loader import MNIST

mndata = MNIST('')

X_train , Y_train = mndata.load_training()
# or
X_test, Y_test = mndata.load_testing()
xp=np.empty((0,784), int)
yp=np.empty((0,1),int)
X_opt=np.empty((0,784), int)
Y_opt=np.empty((0,1),int)
for i in reversed(range(len(Y_train))):
    if Y_train[i] ==1 or Y_train[i] ==0:
        xp=np.append(xp,[X_train[i]],axis=0)
        yp=np.append(yp,[Y_train[i]])

for i in reversed(range(len(Y_test))):
    if Y_test[i] ==1 or Y_test[i] == 0:
        X_opt=np.append(X_opt,[X_test[i]],axis=0)
        Y_opt=np.append(Y_opt,[Y_test[i]])
print("start to optimize")
start = time.time()
error = 0
w = logistic_nmistregression(xp,yp,d)
print("optimization finished")
for i in range(len(Y_opt)):
    x = np.dot(w.T, X_opt[i])
    if x>0:
        if Y_opt[i]==0:
            #print("wx +b=",x,",but y is",ty)
            error = error +1
        else:
            error = error
    if x<0:
        if Y_opt[i]>0:
            error = error +1
        else:
            error = error
print("MNIST error {:.3f} %".format(error/len(Y_opt)*100))
end = time.time()
print("Time elapsed: {:.3f} ".format(end - start),"s")


# In[60]:


d=2
xp,yp,xn,yn=createdata(0)
errort = 0
start = time.time()
for i in range(xp.shape[0]):
    xpp=np.matrix(xp)
    ypp=list(yp)
    testdatax = xpp[i,:]
    testdatay = ypp[i]
    xpp = np.delete(xpp,i,0)
    ypp = np.delete(ypp,i)
    w = logistic_regression(xpp,ypp.T,xn,yn,d,0)
    errort = errort + error( (testdatax @ w), testdatay)
for i in range(xn.shape[0]):
    xnn=np.matrix(xn)
    ynn=list(yn)
    testdatax = xnn[i,:]
    testdatay = ynn[i]
    xnn = np.delete(xnn,i,0)
    ynn = np.delete(ynn,i)
    w = logistic_regression(xp,yp,xnn,ynn.T,d,0)     
    errort = errort + error( (testdatax @ w), testdatay)
print("Logistic regression crossvalidation error:","= {:.3f} %".format(errort/(xp.shape[0]+xn.shape[0])*100))
end = time.time()
print("Time elapsed: {:.3f} ".format(end - start),"s")


# In[24]:


d=2
xp,yp,xn,yn=createdata(0)
beta = logistic_regression(xp,yp,xn,yn,d,1)


# In[ ]:




