# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr as corr
from scipy.stats.stats import linregress as lr

movies = pd.read_csv("C:\\Users\\Jay Ghiya\\Documents\\DataQuest\\Statistics_Beginner_Project\\fandango_score_comparison.csv")
print(movies.head(n=5))
#performing statistical analysis
#let us look at frequency distribution of metacritic vs fandango
plt.hist(movies['Metacritic_norm_round'])
plt.show()
plt.hist(movies['Fandango_Stars'])
plt.show()
"""observations
 As we can see fandango has a minim rating of 3 and maximum of 5
whereas metacritic has minimum rating of 0.5 to 4.5 which is
 more evenly distributed across the scale
 let us get into descriptive statistics - Mean, Median , Standard
 deviation"""
Metacritic_norm_mean = movies['Metacritic_norm_round'].mean()
Fandango_mean = movies['Fandango_Stars'].mean()
Metacritic_norm_median = movies['Metacritic_norm_round'].median()
Fandango_median = movies['Fandango_Stars'].median()
Metacritic_norm_std = movies['Metacritic_norm_round'].std()
Fandango_std = movies['Fandango_Stars'].std()
print("Metacritic mean",Metacritic_norm_mean)
print("Metacritic median",Metacritic_norm_median)
print("Metacritic std",Metacritic_norm_std)
print("Fandango mean",Fandango_mean)
print("Fandango median",Fandango_median)
print("Fandango std",Fandango_std)
""" observation:
    fandango std suggests that data is much closer to mean than metacritic!
    Regarding, mean fandango has a higher mean and median.! 
    let us find corelation between columns with help of scatter plot
also if we subtract metacritic and fandango ratings, we should be able 
   movies with the largest difference which will help us in finding the outliers"""
#scatter plot
plt.scatter(movies["Fandango_Stars"],movies["Metacritic_norm_round"])
plt.show()
""" observation from scatter plot: several movies appear to be low rated in
metacritic but has a high rating in Movie Reviews! let us explore this further"""
#calculating difference between fandango and metacritic columns
movies["fm_diff"] = abs(movies["Metacritic_norm_round"] - movies["Fandango_Stars"])
sort_df = movies.sort_values(by=['fm_diff'],ascending=False)
print(sort_df.head(n=5))
""" observation now we know which movies have highest difference between fandango
and metacritic ! but why? let us try to understand that!
This can be done by finding the corerelation coefficeint first and then we can actually
predict ratings of fandango based on metacritic if those two have a good corerelation coefficient
So, without further due let us get started"""
#calculating corerelation coeefficient first
corr_ratings,p = corr(movies["Fandango_Stars"], movies["Metacritic_norm_round"])
print(type(corr_ratings))
print("corrrelation betwen fandango and metacriticis",corr_ratings)
"""observation  based on corerelation coefficient is it is quite low """
slope,intercept,rvalue,pvalue,stderr = lr(movies["Metacritic_norm_round"],movies["Fandango_Stars"])
#as we have the slope and intercept we should be able to predict the fandango rating based on metacritic
fandango_Rating_prediction_1 = slope * (3.0) + intercept
print(fandango_Rating_prediction_1)
#let us create a residual plot to better visualize metacritic and fandango lines
fandango_Rating_prediction_2 = slope * (1.0) + intercept
fandango_Rating_prediction_3 = slope * (5.0) + intercept
plt.scatter(movies["Metacritic_norm_round"],movies["Fandango_Stars"])
#plotting residuals
list_x = [1.0,5.0]
list_y = [fandango_Rating_prediction_2,fandango_Rating_prediction_3]
plt.plot(list_x,list_y)
plt.xlim(1,5)
plt.show()

