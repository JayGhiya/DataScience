import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error as m_s_e
import matplotlib.pyplot as plt
# predict car's market price using its attributes
#dataset information -> https://archive.ics.uci.edu/ml/datasets/automobile
col_names = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]
cars = pd.read_csv("imports-85.data.csv",names=col_names)
print(cars.columns.values)

#as we can see columns do not have headers. a row needs to be introduced for header
#let us find out, values which can be used in k-nearest neighbours
"""
potential features:
1) normalized losses
2) wheel_base
3) length
4) width
5) height
6) curb-weight
7) engine-size
8) num-of-cylinders
9) engine-size
10) horsepower
11) peak-rpm
12) city-mpg
13) highway-mpg
14)bore
15)stroke
16)Compression ratio

target:
1) price
"""
# Select only the columns with continuous values from - https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
numeric_cars = cars[continuous_values_cols]

#data cleaning
"""no missing values should be there to apply prediction values"""
numeric_cars = numeric_cars.replace({"?":np.nan})
"""all the columns who have numeric values need to be converted into
numeric if they are in string"""
numeric_cars = numeric_cars.astype('float')

numeric_cars = numeric_cars.dropna(subset=['price'])
numeric_cars = numeric_cars.fillna(numeric_cars.mean())
print(numeric_cars.isnull().sum())

#normalizing the columns as the range is different in each column
#do not normalize the target column as it will be used for prediction
price_orig = numeric_cars['price']
numeric_cars = (numeric_cars - numeric_cars.min()) / (numeric_cars.max() - numeric_cars.min())
numeric_cars['price'] = price_orig

#univariate k-nearest neighbors models

def knn_train_test(k,train_colm_names,target_coln_name,df_obj):
    np.random.seed(1)
    shuffled_index = np.random.permutation(df_obj.index)
    rand_df = df_obj.reindex(shuffled_index)
    split_obj = int((len(rand_df) / 2))
    train_set = rand_df[0:split_obj]
    test_set = rand_df[split_obj:]
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(train_set[[train_colm_names]],train_set[target_coln_name])
    predictions = knn.predict(test_set[[train_colm_names]])
    return (np.sqrt(m_s_e(test_set['price'],predictions)))

#determine which column performs the best
rmse_list = []
k_list = [1,3,5,7,9]
col_avg_rmse_list = {}


"""for each in numeric_cars.columns.drop('price'):
    colm_rmse_list = []
    for x in k_list:
        colm_rmse_list.append(knn_train_Test(x,each,'price',numeric_cars))
        rmse_list.append(colm_rmse_list)
            # let us visualize the results using a  scatter plot
    #plt.scatter(k_list,colm_rmse_list)
    #plt.xlabel(each+":k-values")
    #plt.ylabel("Rmse values")
    #plt.show()
    col_avg_rmse_list[each] = np.mean(colm_rmse_list)
#let us build a multivariate model
#here first we need to sort the items in dic based on values
sorted_rmse = pd.Series(col_avg_rmse_list)
#getting the list of columns with lowest rmse values
sorted_rmse = sorted_rmse.sort_values()
print(sorted_rmse)"""

#training the model with 2 fetaures,3 features , 4 features,5 features
k_rmse_results = {}
three_best_features = ['horsepower', 'width', 'curb-weight']
four_best_features = ['horsepower', 'width', 'curb-weight', 'city-mpg']
five_best_features = ['horsepower', 'width', 'curb-weight' , 'city-mpg' , 'highway-mpg']
"""two_best_features = ['horsepower', 'width']
rmse_val = knn_train_test(two_best_features, 'price', numeric_cars)
k_rmse_results["two best features"] = rmse_val


rmse_val = knn_train_test(three_best_features, 'price', numeric_cars)
k_rmse_results["three best features"] = rmse_val

four_best_features = ['horsepower', 'width', 'curb-weight', 'city-mpg']
rmse_val = knn_train_test(four_best_features, 'price', numeric_cars)
k_rmse_results["four best features"] = rmse_val

five_best_features = ['horsepower', 'width', 'curb-weight' , 'city-mpg' , 'highway-mpg']
rmse_val = knn_train_test(five_best_features, 'price', numeric_cars)
k_rmse_results["five best features"] = rmse_val

six_best_features = ['horsepower', 'width', 'curb-weight' , 'city-mpg' , 'highway-mpg', 'length']
rmse_val = knn_train_test(six_best_features, 'price', numeric_cars)
k_rmse_results["six best features"] = rmse_val



Take the top 3 results and let us do hyperparameter tuning for them 
series_rmse = pd.Series(k_rmse_results)
series_rmse = series_rmse.sort_values()
print(series_rmse)
looks like four three and five are the best performing models"""
top_models_list = []
top_models_list.append(three_best_features)
top_models_list.append(four_best_features)
top_models_list.append(five_best_features)

hyperparameter_tuning = range(1,26)
models_k_store = []
for each in top_models_list:
    for x_model_parameters in each:
        k_score_model = {}
        for k in hyperparameter_tuning:
            rmse = knn_train_test(k,x_model_parameters,"price",numeric_cars)
            k_score_model[k] = rmse
        models_k_store.append(k_score_model)

print(models_k_store)
""" what can be done is to find optimal value of k for each model"""



