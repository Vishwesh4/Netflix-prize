import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD


# Data Loading
print("Preparing training files....")
data_folder = '../Data_contest/dataset/'

genome_scores_df=pd.read_csv(data_folder+'genome_scores.csv') # Large (500MB)
movies_df=pd.read_csv(data_folder+'movies.csv')
df_test=pd.read_csv(data_folder+'test.csv') # Large 500MB
df_submission = pd.read_csv(data_folder+'dummy_submission.csv')

df_train_partial = pd.read_csv(data_folder+'train.csv')
df_valid_partial = pd.read_csv(data_folder+'validation.csv')

#Concatenating train and validation set
df_valid_partial=df_valid_partial.drop('timestamp',axis=1)
df_new_train = pd.concat([df_train_partial, df_valid_partial])
df_train_full=df_new_train.reset_index(drop=True)
train_df = df_train_full

train = train_df
test = df_test
test_df= df_test

print('done loading data')

### NORMAL REGRESSION ###
# create movie rating dataset from train
# Feature vector for the 10000 movies, each with a 1128 dimensional vector. 
# If a movie doesn't appear in genome_scores we make it simply the 0 vector.

def generate_XY():
    X=np.zeros((10000,1128)) 

    movies_with_featvecs=set(genome_scores_df['movieId'])
    # The average rating, for each of the movies in the training set. 
    # -1 if it is not in the train set.
    rating_movies = -1*np.ones(10000) 
    # Each movie, is labelled +1 or -1 based on whetherr it is a comedy or not

    for i in range(10000):
        if i not in movies_with_featvecs:
            continue
        temp = genome_scores_df[genome_scores_df['movieId']==i]
        feat_vec= np.array(temp['relevance'])
        X[i,:]=feat_vec


    for i in range(10000):
        temp = train_df[train_df['movieId']==i]
        if len(temp)==0:
            continue
        ratings_curr_movies = temp['rating']
        rating_movies[i] = np.mean(ratings_curr_movies)


    all_genres = []
    for i in range(10000):
        temp = movies_df[movies_df['movieId']==i]
        if len(temp)==0:
            continue
        temp = temp['genres'].values[0]
        temp = temp.split('|')
        for genre in temp:
            if genre not in all_genres:
                all_genres.append(genre)

    X_genre = np.zeros((10000,19))

    for i in range(10000):
        temp = movies_df[movies_df['movieId']==i]
        if len(temp)==0:
            continue
        temp = temp['genres'].values[0]
        temp = temp.split('|')

        for idx, genre in enumerate(all_genres):
            X_genre[i,idx] = genre in temp

    X_concat = np.concatenate((X,X_genre),axis=1)
    return X_concat, rating_movies

def SVR_Predictions(X, rating_movies):
    X_all = X[rating_movies>0]
    Y_all = rating_movies[rating_movies>0]

    best_kernel_param = 0.1
    best_reg_param = 10

    SVM_algo   = SVR(C=best_reg_param, kernel='rbf', gamma = best_kernel_param)
    classifier = SVM_algo.fit(X_all,Y_all)

    X_all_full = X
    Y_pred_all = classifier.predict(X_all_full)
    return Y_pred_all


X, rating_movies = generate_XY()
print('done generation of X')
Y_pred_all = SVR_Predictions(X, rating_movies)
print('done SVR Predictions')

### USER BASED REGRESSION ###
def userbased_regression():
    kernel_param = 0.1
    C = 10
    alpha = 1/(2*C)

    #For user specific
    rating_pred = np.zeros((10000,10000))

    for userId in range(10000):
        User_specific = train_df.loc[train_df['userId'] == userId]
        User_specific_test = test_df.loc[test_df['userId'] == userId]
        if (len(User_specific)!=0) and (len(User_specific_test)!=0):
            X_training_matrix = X[User_specific.movieId,:]
            Y_training_matrix = User_specific.rating
            X_testing_matrix = X[User_specific_test.movieId,:]
            list_movieId = User_specific_test.movieId
            SVM_algo =  KRR(kernel='rbf')
            classifier = SVM_algo.fit(X_training_matrix,Y_training_matrix)
            Y_test_pred_matrix = classifier.predict(X_testing_matrix)
            rating_pred[userId,list_movieId.values] = Y_test_pred_matrix
    return rating_pred


rating_pred = userbased_regression()

# rectifying zero values of user regression values
user_regression = np.zeros(len(df_test))
for i in range(len(df_test)):
    userid =  df_test.iloc[i,0]
    movieid = df_test.iloc[i,1]
    #movie_based
    rating_movie = Y_pred_all[movieid]
    #user based
    rating_user = rating_pred[userid,movieid]
    if rating_user==0:
        rating_user = rating_movie
    user_regression[i] = rating_user



del train_df
del rating_pred
del X


### TRUNCSVD ###
# ADD HERE

#Due to adding of validation, we have some duplicates
movie_matrix = pd.concat([train,test]).drop_duplicates(subset = ['userId','movieId'],keep = 'first')
#Creates a movie matrix of #numofusers vs #noofmovies
movie_matrix = movie_matrix.pivot('userId','movieId','rating')

movie_means = movie_matrix.mean()
user_means = movie_matrix.mean(axis=1)
#Mean shifting
movie_shifted_temp = movie_matrix-movie_means
movie_shifted = movie_shifted_temp.fillna(0)
#To get locations where we have ratings
mask = -movie_shifted_temp.isnull()

def repeated_matrix_reconstruction(num_pcs,num_iterations):
    global movie_shifted
    for i in range(num_iterations):
        SVD = TruncatedSVD(n_components=num_pcs,random_state=42)
        SVD.fit(movie_shifted)
        #For the ease of applying masks we work with pandas
        movie_represented =  pd.DataFrame(SVD.inverse_transform(SVD.transform(movie_shifted)),columns=movie_shifted.columns,index=movie_shifted.index)
        loss = mean_squared_error(movie_represented[mask].fillna(0),movie_shifted_temp[mask].fillna(0))
        print('Iteration: {} , Loss: {} '.format(i,loss))
        #To just update the non-zero values of movie_reprented values to the true ratings
        movie_represented[mask] = movie_shifted_temp[mask]
        movie_shifted = movie_represented
    #Mean shifting it back
    movie_mat = movie_shifted + movie_means
    movie_mat = movie_mat.clip(lower=0.5,upper=5)
    return movie_mat
print("Starting truncated svd with number of components as 20")
representative_matrix_20 = repeated_matrix_reconstruction(20,10)
print("Done")
print("Starting truncated svd with number of components as 15")
representative_matrix_15 = repeated_matrix_reconstruction(15,10)
print("Done")
#bagging
rating_matrix = (representative_matrix_15+representative_matrix_20)/2


trunc_prediction = np.zeros(len(test))
for i in range(len(test)):
    userid =  test.iloc[i,0]
    movieid = test.iloc[i,1]
    trunc_prediction[i] = rating_matrix[rating_matrix.index==userid][movieid].values[0]
    
indices=np.argwhere(np.isnan(trunc_prediction))
trunc_prediction[indices] = user_regression[indices]


# ENSEMBLING
PRED = (2*trunc_prediction + 1*user_regression)/3  # best 2:1
PRED = np.around(PRED,1)

PRED = np.clip(PRED, a_min = 0.5, a_max = 5)
# SUBMISSION

df_submission.Prediction = PRED
df_submission.to_csv('./Goodfellas_submission.csv',index=False)

print('Done!!!')




