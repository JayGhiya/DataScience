import pandas as pd
import re
import numpy as np
from scipy.stats import chisquare

jeopardy_data = pd.read_csv("jeopardy.csv")
print(jeopardy_data.columns.values)
#found spaces in column names!let us remove them
jeopardy_data.columns =[x.replace(" ","") for x in jeopardy_data.columns.values]
print(jeopardy_data.columns.values)
def normalize_questions_answers(un_item):
    un_item = str(un_item).lower()
    un_item = re.sub("[^A-Za-z0-9\s]", "", un_item)
    return un_item

jeopardy_data['clean_question'] = jeopardy_data['Question'].apply(normalize_questions_answers)

jeopardy_data['clean_answer'] = jeopardy_data['Answer'].apply(normalize_questions_answers)

def normalize_dollar_Column(un_item):
    un_item = re.sub("[^A-Za-z0-9\s]", "", un_item)
    try:
        un_item = int(un_item)
    except ValueError:
        un_item = 0
    return un_item

jeopardy_data['clean_value'] = jeopardy_data['Value'].apply(normalize_dollar_Column)

jeopardy_data['AirDate'] = pd.to_datetime(jeopardy_data['AirDate'])

"""let us find out how often the answer is deducible from the question"""
def deduce_ans_from_q(un_series):
    split_answer = un_series['clean_answer'].split(" ")
    split_question = un_series['clean_question'].split(" ")
    if "the" in split_answer:
        split_answer.remove("the")
    if len(split_answer) == 0:
        return 0
    match_count = 0
    for item in split_answer:
        if item in split_question:
            match_count += 1
    return (match_count/ (len(split_answer)))


jeopardy_data['answer_in_question'] = jeopardy_data.apply(deduce_ans_from_q,axis=1)
print(jeopardy_data['answer_in_question'].mean())
#only 6% percent of answer appears in the question
""" let us find out number of questions which get repeated """
#first let us sort the dataframe by air date
jeopardy_data = jeopardy_data.sort_values(by=['AirDate'])
question_overlap = []
terms_used = set()

for index,row in jeopardy_data.iterrows():
    split_question = row['clean_question'].split(" ")
    split_question = [x for x in split_question if len(x) > 5]
    match_count = 0
    for word in split_question:
        if word in terms_used:
            match_count+=1
        terms_used.add(word)
    if len(split_question) > 0:
        match_count = match_count / len(split_question)
    question_overlap.append(match_count)

print(np.mean(question_overlap))
#70 % percent of time questions are overlapped ! we need to look into this further
""" let us focus on questions with high values ! to do that we need to divide the questions in small ones and big ones"""
#let us break down into two categories: high value and low value

def divide_categories(row):
    value = int()
    if row['clean_value'] > 800:
        value = 1
    else:
        value = 0
    return value

jeopardy_data['high_value'] = jeopardy_data.apply(divide_categories,axis=1)

def calculate_word_worth(word):
    low_count = 0
    high_count = 0
    for index,row in jeopardy_data.iterrows():
        split_question = row['clean_question'].split(" ")
        if word in split_question:
            if row['high_value'] == 1:
                high_count += 1
            else:
                low_count += 1
    return (high_count,low_count)


observed_expectecd = []
# we will working on a small sample let us take out 5 terms from our main list

comparison_terms = list(terms_used)[:5]

for word in comparison_terms:
    high_and_low_value = calculate_word_worth(word)
    observed_expectecd.append(high_and_low_value)

print(observed_expectecd)
#let us find the expected items
high_value_count = jeopardy_data[jeopardy_data["high_value"] == 1].shape[0]
low_value_count = jeopardy_data[jeopardy_data["high_value"] == 0].shape[0]

chi_squared = []

for total_Count in observed_expectecd:
    total = total_Count[0] + total_Count[1]
    total_prop = total / jeopardy_data.shape[0]
    expected_high_value_count = total_prop * high_value_count
    expected_low_value_count = total_prop * low_value_count
    obs = np.array([total_Count[0],total_Count[1]])
    exp = np.array([expected_high_value_count,expected_low_value_count])
    chi_squared.append(chisquare(obs,exp))

print(chi_squared)


#there is no significant difference in usage of terms between observed and expected values and also p value is quite high. It is suggesting it may be due to chance. But the data set is quite small and has low
# frequencies. Terms with large frequencies should be tested using a chi squared test

















