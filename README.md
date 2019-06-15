# Netflix Data Challenge
This was a classroom data contest hosted on Kaggle by IIT Madras Computer Science Department as PRML data contest. Please follow this link for more details:- [Contest link](https://www.kaggle.com/c/prml19/data)
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Libraries needed

```
Pandas
Numpy
Sklearn 
```
## Introduction:
There are several methods to solve the movielens data contest. In our documentation, we
explore different methods used to tackle the problem, and comment on what worked the best
and what didn’t work.
## Models which didn't work
We tried a lot of models and a major portion of them didn't work.
- Neighbourhood model: the score improvement that was given by this model was not
  worth the computation effort it was taking
- Neighbourhood model on regression instead of baseline : this also didn't give as much results as the algorithm was simple over fitting and it's hypothesis didn't make sense as the neighbourhood model on baseline did
- Neighbourhood model on ensemble of regression and baseline
- Neighbourhood model on ensemble of latent factor model
- Latent factor on ensemble of regression and baseline
- Latent factor on regression
- Latent factor on neighbourhood model
- User regression and regression on movies ensemble: This probably happened because user regression is a local and smaller    version of movie regression. Hence we didn't get a good improvement from before.<br><br>
These model didn't give much improvement , we also tried other models which we didn't include in the end because we got  better results using other methods
- Neural networks regression
- Latent factor
- Baseline
## Models which worked:
- Repeated matrix reconstruction
- Regression on movies
- Regression on users<br>
We realized from our tries that ensembles works well only if your models are independent of 
each other and if it tackles the problem in a different way. Hence developing models using
different methods helped us in reducing our MSE error.<br><br>
<u><b>Repeated Matrix Reconstruction[1]</b></u><br>
Consider the ratings matrix X, with rows as User Ids and columns as Movie Ids. A natural  way to
tackle this problem is to group users and movies into genres, like we did in latent factor model.
Based on this similar idea, we believe that the matrix X consists of a lower rank, hence low rank
approximation is well suited for problem such as this. Here, the author attempts to decompose
the matrix using SVD into smaller rank matrices<br>
X = U SV .T , which is analogous to user groups and movie genres<br><br>
<u><b>Pseudo-code:</b></u><br>
- Preprocessing step
  - First generate a rating matrix X from both train and test file
  - Mean shift the rating matrix X w.r.t rated entries
  - Make the non-rated entries as 0. These are the entries which we will procure by USV.T<br><br>
- <u><b>Repeated matrix reconstruction</b></u><br>
  - Using truncated SVD, get the new matrix X with n-principle components. This is
got by doing SVD and selecting the top n eigenvectors of V, U and n eigenvalues
S and take the product as ,Xnew = U1S1V1.T
  - Since this approximation is not guaranteed to keep the original rated entries as
same, we reset those values to the original rating, and repeat step 1 again, but
without any change in the non-rated entries
  - If there is truly a lower rank matrix, this algorithm should converge
  - Mean shift it back to get the ratings
  - There are movies which no user rated in train, we fill those values using
regression values mentioned below.<br><br>

Based on the results of research paper[1] which we followed, we ensemble models with number 
of PCs-15 and with number of PCs-20<br><br>

Just using the model, we got<b> MSE = 0.76</b> <br><br>

<b><u>Regression on movies:</u></b><br>
In our algorithms for baseline, neighbourhood models or latent factors , we are not using the
information about the movies. In essence of collaborative filtering , we are just trying to group
movies into genres and users into groups for most of the algorithms. For instance, just using the
information that the movie has Tom Hanks or it is made by Steven Spielberg we can surely say
that the movie will perform above average. Here we take advantage of the different informations
that is provided for each movies given in the genome.csv. We also included the genres as
features. We ignored the tags.csv here because based on the data provided in the movies, most
of the input felt like noise.<br><br>
-<b><u>Psuedocode:</u></b><br>
 - Build the movie matrix by taking a movie and building its features from
genome.csv.
 - Build Y vector as the average of all users for each movies. For movies not rated ,
put ratings as -1
 - Build SVR model using the rated movies from movie matrix and Y vector.<br><br>
Just using the model, we got​ <b>MSE = 0.79</b><br><br>
-<b><u>Regression on users:</u></b><br>
We build on the top of regression on movies. We now also consider the effect of users. Since
we don’t have any user specific knowledge, we build a separate model for each user. For each
user, we find the movies rated by that user and using the features that we procured earlier, we
build a user specific training matrix and build a model for that user. The procedure is same for
training.<br>
Since we encountered a lot of users in test but not in train, we replaced those values with
regression on movies.<br><br>
- <b><u>Psuedocode:</b></u><br>
 - For each UserId, create a matrix of movies rated by that user with each row
containing the 1128 features of the movie.
 - Pass this matrix through a regressor, We used Kernel Ridge Regressor from
sklearn. The kernel used was ‘rbf’. Regularization was set to auto.
 - Since each of the model required a lot of memory to store, we decided not to
store the models.
 - Instead, for each userId, after fitting the model immediately predicted on the test
dataset for all the entries containing that userId.
 - The ratings which remained zero after step 4 are replaced by the ratings from
movie based regression model.<br><br>
Just using the model, we got <b>MSE = 0.76</b><br>

## End model
The final prediction is the ensemble of user regression and repeated matrix reconstruction
.Based on the leaderboard scores, we improved our scores by varying the weights associated to
each model.<br>
Final Prediction = <i>(2*Pred_repeated_matrix_reconstruction + User_regression)/3</i><br>
<b>Final MSE = 0.728</b>
## Running the tests
You can download the data from the kaggle link provided above.
We have some important files
* `1 - FINAL_SUBMISSION_CODE_Goodfellas(1).py` - Contains the final model used for submission 
* `2 - Prml_datacontest.ipynb` - contains Neighbourhood model
* `3 - data.pickle` - contains saved movie based regression model
* `4 - latent_vector_model.ipynb` - contains latent vector model
* `5 - saved_models` - contains the saved checkpoints of both prediction and model weights
* `6 - truncated_svd.ipynb` - contains the truncated SVD model
## References
- http://cs229.stanford.edu/proj2006/KleemanDenuitHenderson-MatrixFactorizationForCollaborativePrediction.pdf
- http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/
- <b>Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering
Model</b> by Yehuda Koren
- <b>The BellKor solution to the Netflix Prize</b> by Robert M. Bell, Yehuda Koren and Chris
Volinsky.

