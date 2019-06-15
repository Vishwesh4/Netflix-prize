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
- Neighbourhood model on regression instead of baseline​ : this also didn't give as much<br>
  results as the algorithm was simple over fitting and it's hypothesis didn't make sense as<br>
  the neighbourhood model on baseline did<br>
- Neighbourhood model on ensemble of regression and baseline
- Neighbourhood model on ensemble of latent factor model
- Latent factor on ensemble of regression and baseline
- Latent factor on regression
- Latent factor on neighbourhood model
- User regression and regression on movies ensemble: ​ This probably happened because<br>
  user regression is a local and smaller version of movie regression. Hence we didn't get<br>
  a good improvement from before.<br>
These model didn't give much improvement , we also tried other models which we didn't include in the end because we got<br> better results using other methods<br>
- Neural networks regression
- Latent factor
- Baseline
## Models which worked:
- Repeated matrix reconstruction
- Regression on movies
- Regression on users
We realized from our tries that ensembles works well only if your models are independent of<br>
each other and if it tackles the problem in a different way. Hence developing models using<br>
different methods helped us in reducing our MSE error.<br>
Repeated Matrix Reconstruction[1]Consider the ratings matrix X, with rows as User Ids and columns as Movie Ids. A natural<br> way to
tackle this problem is to group users and movies into genres, like we did in latent factor model.<br>
Based on this similar idea, we believe that the matrix X consists of a lower rank, hence low rank<br>
approximation is well suited for problem such as this. Here, the author attempts to decompose<br>
the matrix using SVD into smaller rank matrices<br>
X = U SV .T , which is analogous to user groups and movie genres<br>
<u><b>Pseudo-code:</b></u>
-Preprocessing step
 - First generate a rating matrix X from both train and test file
 - Mean shift the rating matrix X w.r.t rated entries
 - Make the non-rated entries as 0. These are the entries which we will procure by USV.T
<u><b>Repeated matrix reconstruction</b></u>
 - Using truncated SVD, get the new matrix X with n-principle components. This is<br>
got by doing SVD and selecting the top n eigenvectors of V, U and n eigenvalues<br>
S and take the product as ,Xnew = U1S1V1.T
 - Since this approximation is not guaranteed to keep the original rated entries as
same, we reset those values to the original rating, and repeat step 1 again, but
without any change in the non-rated entries
 - If there is truly a lower rank matrix, this algorithm should converge
 - Mean shift it back to get the ratings
 - There are movies which no user rated in train, we fill those values using<br>
regression values mentioned below.<br>

Based on the results of research paper[1] which we followed, we ensemble models with number<br>
of PCs-15 and with number of PCs-20<br>

Just using the model, we got​ MSE = 0.76

<b><u>Regression on movies:</u></b>
In our algorithms for baseline, neighbourhood models or latent factors , we are not using the<br>
information about the movies. In essence of collaborative filtering , we are just trying to group<br>
movies into genres and users into groups for most of the algorithms. For instance, just using the<br>
information that the movie has Tom Hanks or it is made by Steven Spielberg we can surely say<br>
that the movie will perform above average. Here we take advantage of the different informations<br>
that is provided for each movies given in the genome.csv. We also included the genres as<br>
features. We ignored the tags.csv here because based on the data provided in the movies, most<br>
of the input felt like noise.<br>
<b><u>Psuedocode:</u></b>
- Build the movie matrix by taking a movie and building its features from
genome.csv.
- Build Y vector as the average of all users for each movies. For movies not rated ,
put ratings as -1
- Build SVR model using the rated movies from movie matrix and Y vector.<br>
Just using the model, we got​ <b>MSE = 0.79</b>
<b><u>Regression on users:</u></b>
We build on the top of regression on movies. We now also consider the effect of users. Since<br>
we don’t have any user specific knowledge, we build a separate model for each user. For each<br>
user, we find the movies rated by that user and using the features that we procured earlier, we<br>
build a user specific training matrix and build a model for that user. The procedure is same for<br>
training.<br>
Since we encountered a lot of users in test but not in train, we replaced those values with<br>
regression on movies.<br>
<b><u>Psuedocode:</b></u>
- For each UserId, ​ create a matrix of movies rated by that user with each row
containing the 1128 features of the movie.
- Pass this matrix through a regressor, We used Kernel Ridge Regressor from
sklearn. The kernel used was ‘rbf’. Regularization was set to auto.
- Since each of the model required a lot of memory to store, we decided not to
store the models.
- Instead, for each userId, after fitting the model immediately predicted on the test
dataset for all the entries containing that userId.
- The ratings which remained zero after step 4 are replaced by the ratings from
movie based regression model.
Just using the model, we got​ <b>MSE = 0.76</b>

## End model
The final prediction is the ensemble of user regression and repeated matrix reconstruction<br>
.Based on the leaderboard scores, we improved our scores by varying the weights associated to<br>
each model.<br>
Final Prediction = (2*Pred_repeated_matrix_reconstruction + User_regression)/3
<b>Final MSE = 0.728</b>
## Running the tests
You can load data from :-
* `1 - intraday/quote_in.csv` - data for quotes, training set
* `2 - intraday/quote_out.csv` - data for quotes, test set
* `3 - intraday/trade_in.csv` - trade data , training set
* `4 - intraday/trade_out.csv` - trade data , test set

We have some important files
* `1 - model.ipynb` - Main pipeline
* `2 - core/model.py` - contains class model which is our LSTM model
* `3 - core/data_processor.py` - data_loader class for ease of loading big data
* `4 - config.json` - contains all the important parameters for model
* `5 - saved_models` - contains the saved checkpoints of both prediction and model weights
* `6 - model.h5` - contains the current model weights we are using for replicability
