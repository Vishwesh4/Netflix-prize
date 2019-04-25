import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

#Loading dataset
print("Preparing training files....")
df_train_partial = pd.read_csv("./dataset/train.csv")
df_valid_partial = pd.read_csv("./dataset/validation.csv")
df_test = pd.read_csv("./dataset/test.csv")
df_submission = pd.read_csv("./dataset/dummy_submission.csv")

#Concatenating train and validation set
df_valid_partial=df_valid_partial.drop('timestamp',axis=1)
df_new_train = pd.concat([df_train_partial, df_valid_partial])
df_train_full=df_new_train.reset_index(drop=True)
print("Done")

def minibatch(batch_no,df,batch_size,indices):
    '''Generates the batch for mini-batch gradient descent '''
    if (batch_no+1)*batch_size > len(df):
        minibatch_indices = indices[batch_size*(batch_no)::]
    else:
        minibatch_indices = indices[batch_size*(batch_no):batch_size*(batch_no+1)]
    X_train = df.iloc[minibatch_indices,:]
    X_train = X_train.sort_values(by=['userId','movieId']).reset_index(drop=True)
    return X_train


#baseline
def loss_baseline(parameters,df):
    miu,bu,bi = parameters
    loss = 0
    for i in range(len(df)):
        userid =  df.iloc[i,0]
        movieid = df.iloc[i,1]
        rating = miu+bu[userid]+bi[movieid]
        if rating>5:
            rating = 5
        if rating<0.5:
            rating =0.5
        loss = loss + (df.iloc[i,2]-(miu+bu[userid]+bi[movieid]))**2
    loss = loss/len(df)
    return loss

def Baseline_Training(step_size,lamda,Epochs,batch_size,df):
    '''Does Mini-batch Gradient descent to optimize baseline training error'''

    np.random.seed(5)
    indices = np.random.permutation(df.shape[0])

    #Initialize values for bu,b'i,miu
    #Since we are given there are 10k users and 10k movie id
    bu = np.zeros(10000)
    bi = np.zeros(10000)
    #Taking initial guess of miu as average of all ratings
    miu = np.mean(df.iloc[:,2])

    num_batches = df.shape[0]//batch_size + 1
    for epoch in range(Epochs):
        for batch in range(num_batches):
            bui = []
            X = minibatch(batch,df,batch_size,indices)
            for i in range(len(X)):
                userid =  X.iloc[i,0]
                movieid = X.iloc[i,1]
                bui.append((miu+bu[userid]+bi[movieid]))
            #Adding the regularization term
            bu[X.userId.unique()] = (1-step_size*lamda)*bu[X.userId.unique()]
            bi[X.movieId.unique()] = (1-step_size*lamda)*bi[X.movieId.unique()]
            miu = (1-step_size*lamda)*miu
            for i in range(len(X)):
                userid =  X.iloc[i,0]
                movieid = X.iloc[i,1]
                bu[userid] = bu[userid] + step_size*(X.iloc[i,2]-bui[i])
                bi[movieid] = bi[movieid] + step_size*(X.iloc[i,2]-bui[i])
                miu = miu + step_size*(X.iloc[i,2]-bui[i])
            print("Epoch:{} Batch:{} ------ Training Error:{}".format(epoch,batch,loss_baseline((miu,bu,bi),X)))
    return miu,bu,bi
import time
start = time.time()
#Training baseline
print("Training Baseline....")
#Hyperparameters
step_size = 0.001
lamda = 0
Epochs = 1
batch_size = 1024
#Training baseline model
miu,bu,bi = Baseline_Training(step_size,lamda,Epochs,batch_size,df_train_full)
print("Done")
end = time.time()
print(end-start)
def baseline(movieid,userid):
    return (mui+bu[userid]+bi[movieid])

#latent model
def loss_latent(params,df):
    pu,qi = params
    loss = 0
    for i in range(len(df)):
        userid =  df.iloc[i,0]
        movieid = df.iloc[i,1]
        rating = baseline(movieid,userid)+np.dot(pu[userid],qi[movieid])
        if rating>5:
            rating = 5
        if rating<0.5:
            rating =0.5
        loss = loss + (df.iloc[i,2]-rating)**2
    loss = loss/len(df)
    return loss

def Latent_Training_with_baseline(step_size,dim,Epochs,batch_size,df):
    #Since we are given there are 10k users and 10k movie id
    np.random.seed(5)
    indices = np.random.permutation(df.shape[0])

    pu = np.ones((10000,dim)) / dim
    qi = np.ones((10000,dim)) / dim
    num_batches = df.shape[0]//batch_size + 1
    for epoch in range(Epochs):
        for batch in range(num_batches):
            X = minibatch(batch,df,batch_size,indices)
            userid =  X.iloc[:,0]
            movieid = X.iloc[:,1]
            bui = baseline(movieid,userid)
            pu_temp = pu
            qi_temp = qi
            for i in range(len(X)):
                userid =  X.iloc[i,0]
                movieid = X.iloc[i,1]
                prod = np.dot(pu[userid],qi[movieid])
                pu_temp[userid]  += step_size*qi[movieid]*(X.iloc[i,2]-bui[i]-prod)
                qi_temp[movieid] += step_size*pu[userid]*(X.iloc[i,2]-bui[i]-prod)
            pu = pu_temp
            qi = qi_temp

            print("Epoch:{} Batch:{} ------ Training Error:{}".format(epoch,batch,loss_latent((pu,qi),X)))
    return pu, qi

#Training Latent
print("Training Latent model ....")

#Hyperparameters
step_size = 0.01
dim = 50
Epochs = 1
batch_size = 2048

#latent model training
pu, qi = Latent_Training_with_baseline(step_size,dim,Epochs,batch_size,df_train_full)
print("Done")

#regression and Neural Network regression

#Truncsvd


#Submission file
predictions = np.zeros(len(df_test))
for i in range(len(df_test)):
    userid =  df_test.iloc[i,0]
    movieid = df_test.iloc[i,1]
    #ratings
    rating_latent = baseline(movieid,userid)+np.dot(pu[userid],qi[movieid])
    rating_svr = Y_pred_SVR[movieid]
    rating_mlpr = Y_pred_MLPR[movieid]

    rating = float("{0:.1f}".format((8*rating_trunc + rating_latent + rating_svr + 4*rating_mlpr)/14))
    if rating>5:
        rating = 5
    if rating<0.5:
        rating =0.5
    predictions[i] = rating
df_submission.Prediction = predictions
df_submission.to_csv('./Submissions/Submission_goodfellas.csv',index=False)
