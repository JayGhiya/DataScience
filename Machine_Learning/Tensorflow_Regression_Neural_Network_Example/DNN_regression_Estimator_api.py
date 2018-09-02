import math


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10


#loading the data

california_housing_df = pd.read_csv("california_housing_train.csv",sep=",")

#shuffling the data to avoid sampling bias
np.random.seed(1)
california_housing_df = california_housing_df.reindex(np.random.permutation(california_housing_df.index))



def preprocess_features(housing_df):
    features = housing_df[["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]

    processed_features = features.copy()

    processed_features["rooms_per_person"] = california_housing_df["total_rooms"] / california_housing_df["population"]

    return processed_features

def preproces_targets(housing_df):
    targets = pd.DataFrame()
    targets["median_house_value"] = (housing_df["median_house_value"] / 1000 )
    return targets

"""Data splitting tips:
1) Randomize the data before splitting so that no sampling bias occurs.
2) Split the data into three parts : 1) Training set 2) Validation set 3) Test set
3) Train the model on Training set
4) Evaluate the model performance on validation set.
5) Tweak the hyper parameters according to model performance until you get good accuracy on Validation set.
6) Then  predict test set and the performance should be the same as validation test
"""

#data splitting into 70 % training data and 30 % validation data
training_examples = preprocess_features(california_housing_df.head(12000))
training_targets = preproces_targets(california_housing_df.head(12000))

validation_examples = preprocess_features(california_housing_df.tail(5000))
validation_targets = preproces_targets(california_housing_df.tail(5000))

#describe the numerical columns
print("Training examples summary;")
print(training_examples.describe())
print("Validation examples summary")
print(validation_examples.describe())
print("Training targets summary")
print(training_targets.describe())
print("Validation targets summary")
print(validation_targets.describe())

"""we will be using default relu actiation function for neurons as it is better than standard sigmoid function"""

"""high level estimator api also needs data type of feature columns! writing a generic function for that """

def define_data_type_for_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(feature) for feature in input_features])

"""defining a generic input function for training and prediction"""

def input_function(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    #converting features into a dictionary which will be used for data set api
    features = {key:np.array(value) for key,value in dict(features).items()}

    ds  = Dataset.from_tensor_slices((features,targets))
    ds= ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(10000)

    #once dataset is constructed , need to get itertor for retrievl! we will be using one-shot iterator

    features,labels = ds.make_one_shot_iterator().get_next()
    return features,labels


#define a training function for regression
#this function should return regression model for trained data
def train_nn_regression_model(learning_rate,steps,batch_size,hidden_units,training_examples,training_targets,validation_examples,validation_targets):
    periods = 10
    steps_per_period = steps / periods

    #define gradient descent optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer,clip_norm=5)

    dnn_regressor = tf.estimator.DNNRegressor(feature_columns=define_data_type_for_feature_columns(training_examples),hidden_units=hidden_units,optimizer=optimizer)

    #creating input functions
    training_input_function = lambda : input_function(training_examples,training_targets["median_house_value"],batch_size=batch_size)
    training_predict_function = lambda : input_function(training_examples,training_targets["median_house_value"],num_epochs=1,shuffle=False)
    validation_predict_function = lambda : input_function(validation_examples,validation_targets["median_house_value"],num_epochs=1,shuffle=False)


    """
    To observe the loss function we will follow the below algotithm :

1) Convert the algorithm into periods.
2) For each period:
    a) Find steps per period : Number of steps / Number of periods
    b) Train 
    c) Predict on training and validation data set
    d) Calculate rmse and add it to list
3) Iterate over the list and get the outputs and plot periods vs rmse. Should be a downhill.
    """
    print("Starting periodic training and evaluation")

    training_rmse = list()
    validation_rmse = list()

    for s_p in range(0,periods):
        dnn_regressor.train(input_fn=training_input_function,steps=steps_per_period)
        #calculate predictions For training examples
        training_results = dnn_regressor.predict(input_fn=training_predict_function)
        #converting the result into one dimensional numpy array
        training_results = np.array([item["predictions"][0] for item in training_results])
        #calculate predictions for validation examples
        validation_results = dnn_regressor.predict(input_fn=validation_predict_function)
        validation_results = np.array([item["predictions"][0] for item in validation_results ])
        curr_training_rmse = math.sqrt(metrics.mean_squared_error(training_results,training_targets))
        curr_validation_rmse = math.sqrt(metrics.mean_squared_error(validation_results,validation_targets))
        training_rmse.append(curr_training_rmse)
        validation_rmse.append(curr_validation_rmse)

#plot the loss function vs periods! there should be a downward trend!
    plt.xlabel("Periods")
    plt.ylabel("RMSE")

    plt.title("Root Mean Squarede Error vs. Periods")
    plt.plot(training_rmse,label="training")
    plt.plot(validation_rmse,label="validation")
    plt.legend()
    plt.show()
    return dnn_regressor,training_rmse,validation_rmse

#training the actual model

dnn_regressor,training_rmse,validation_rmse = train_nn_regression_model(learning_rate=0.001,
    steps=2000,
    batch_size=100,
    hidden_units=[10, 10],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

print("minimum training rmse value",min(training_rmse))
print("minimum validation rmse value",min(validation_rmse))

#testing on TEST DATA
housing_test_data = pd.read_csv("california_housing_test.csv", sep=",")

"getting features and labels of test data"
test_features = preprocess_features(housing_test_data)
test_targets  = preproces_targets(housing_test_data)
test_predict_function = lambda : input_function(test_features,test_targets["median_house_value"],num_epochs=1,shuffle=False)
test_results = dnn_regressor.predict(input_fn=test_predict_function)

test_results = np.array([item["predictions"][0] for item in test_results])
print("test rmse",math.sqrt(metrics.mean_squared_error(test_results,test_targets)))



