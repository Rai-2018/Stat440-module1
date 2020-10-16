# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:41:12 2020

@author: wlian
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:00:17 2020

@author: wlian
"""

import pandas as pd
import datetime
import re
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel

from rfpimp import permutation_importances
from sklearn.metrics import r2_score


train = pd.read_csv("train2.txt")
test = pd.read_csv("test2.txt")

train = train.fillna("")
test = test.fillna("")

full_table = pd.concat([train,test])

def parse_symptoms(df):
    if re.search("fever",df.symptoms) and re.search("cough",df.symptoms):
        val = 4
    elif re.search("cough",df.symptoms):
        val = 3
    elif re.search("fever",df.symptoms):
        val = 2
    elif df.symptoms == "":
        val = 0
    else:
        val = 1
    return val

def parse_age(df):
    if df.age == "":
        val = None
    elif re.search("-",df.age):
        parsed = df.age.split("-")
        parsed = list(map(int, parsed))
        val = sum(parsed)/2
        return val
    else:
        val = float(df.age)
    return val

def table_freq(df,col):
    name = col+"_new"
    count = df.groupby([col]).size().reset_index(name=name)
    return count

def parse_freq(df,col):
    return pd.merge(df, table, left_on=col, right_on=col)

def table_freq_pairwise(df,col1,col2):
    name = col1 + "_" + col2
    count = df.groupby([col1, col2]).size().reset_index(name=name)
    return count

def parse_freq_pairwise(df,col1,col2):
    return pd.merge(df, table, left_on=[col1,col2], right_on = [col1,col2])

##############################################################
full_table['maj_symptoms'] = full_table.apply(parse_symptoms, axis=1)

table = table_freq(full_table, "sex")
full_table = parse_freq(full_table, "sex")

table = table_freq(full_table, "country")
full_table = parse_freq(full_table, "country")

table = table_freq(full_table, "V1")
full_table = parse_freq(full_table, "V1")

table = table_freq(full_table, "confirmed")
full_table = parse_freq(full_table, "confirmed")

table = table_freq_pairwise(full_table, "city","province")
full_table = parse_freq_pairwise(full_table, "city","province")

table = table_freq_pairwise(full_table, "province","country")
full_table = parse_freq_pairwise(full_table, "province","country")


full_table["age_new"] = full_table.apply(parse_age, axis=1)
mean_age = full_table["age_new"].mean(skipna=True)
full_table["age_new"] = full_table["age_new"].fillna(mean_age)


full_table['confirmed_dt'] = pd.to_datetime(full_table.confirmed,format="%d.%m.%Y",errors='coerce')

full_table["sex"] = full_table["sex"].astype('category')
full_table["city"] = full_table["city"].astype('category')
full_table["province"] = full_table["province"].astype('category')
full_table["country"] = full_table["country"].astype('category')
full_table["V1"] = full_table["V1"].astype('category')
full_table["confirmed"] = full_table["confirmed"].astype('category')
full_table["outcome"] = full_table["outcome"].astype('category')
full_table["maj_symptoms"] = full_table["maj_symptoms"].astype('category')

full_table["sex_cat"] = full_table["sex"].cat.codes
full_table["city_cat"] = full_table["city"].cat.codes
full_table["province_cat"] = full_table["province"].cat.codes
full_table["country_cat"] = full_table["country"].cat.codes
full_table["V1_cat"] = full_table["V1"].cat.codes
full_table["confirmed_cat"] = full_table["confirmed"].cat.codes
full_table["outcome_cat"] = full_table["outcome"].cat.codes


full_table_onehot = pd.get_dummies(full_table, columns=["sex_cat", "city_cat","province_cat", "country_cat",
                                    "V1_cat", "confirmed_cat","outcome_cat", "maj_symptoms"], 
                                   prefix=["sex_cat", "city_cat","province_cat", "country_cat",
                                           "V1_cat", "confirmed_cat","outcome_cat", "maj_symptoms"],drop_first=True)

train = full_table_onehot[np.isnan(full_table.Id)]
test = full_table_onehot[pd.notnull(full_table.Id)]
# print(full_table.dtypes,"\n")


##############################################################
# Correlation

# plt.matshow(train.corr())
# plt.yticks(range(len(train_select.columns)), train_select.columns)
# plt.colorbar()
# plt.show()
##############################################################


# Setting up Training dataset 
drop = ["age","sex","city","province",
        "country","V1","confirmed", "symptoms",
        "outcome","duration","Id", "sex_new",
        "country_new","V1_new","confirmed_new",
        "city_province","province_country" ,"confirmed_dt"]

train_select = train
train_select_y = train["duration"]
train_select_x = train_select.drop(drop,axis=1)

train_features, test_features, train_labels, test_labels = train_test_split(train_select_x, train_select_y, test_size = 0.20, random_state = 42)

##############################################################
# Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10,15,20,25]
# min_samples_leaf = [1, 2, 4, 6, 8, 10]
# bootstrap = [True, False]
# random_grid = {'n_estimators': n_estimators,
#                 'max_features': max_features,
#                 'max_depth': max_depth,
#                 'min_samples_split': min_samples_split,
#                 'min_samples_leaf': min_samples_leaf,
#                 'bootstrap': bootstrap}



# rf = RandomForestRegressor()
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 500, cv = 10, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(train_features, train_labels)


# print(rf_random.best_params_)
##############################################################



rf = RandomForestRegressor(n_estimators=200,min_samples_split= 10, min_samples_leaf= 1, max_features= 'sqrt', max_depth= 100, bootstrap= True,oob_score=(True))
rf.fit(train_features,train_labels)
print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}\n'.format(rf.score(train_features, train_labels), 
                                                                                              rf.oob_score_,
                                                                                              rf.score(test_features, test_labels)))

def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))

perm_imp_rfpimp = permutation_importances(rf, train_features, train_labels, r2)
perm_imp_rfpimp.reset_index(level=0, inplace=True)

perm_imp_rfpimp = perm_imp_rfpimp[perm_imp_rfpimp.Importance > 0.0001]
perm_imp_rfpimp = perm_imp_rfpimp.Feature
train_select_x = train_select_x[train_select_x.columns & perm_imp_rfpimp]


train_features, test_features, train_labels, test_labels = train_test_split(train_select_x, train_select_y, test_size = 0.20, random_state = 42)
rf = RandomForestRegressor(n_estimators=200,min_samples_split= 10, min_samples_leaf= 1, max_features= 'sqrt', max_depth= 100, bootstrap= True)
rf.fit(train_features,train_labels)
print('R^2 Training Score: {:.2f} \nR^2 Validation Score: {:.2f}\n'.format(rf.score(train_features, train_labels), 
                                                                            rf.score(test_features, test_labels)))


rf = RandomForestRegressor(n_estimators=200,min_samples_split= 10, min_samples_leaf= 1, max_features= 'sqrt', max_depth= 100, bootstrap= True)
rf.fit(train_select_x,train_select_y)
print('R^2 Training Score: {:.2f} \n'.format(rf.score(train_select_x, train_select_y)))





##############################################################

# Validation
test_select = test
test_select_id = pd.DataFrame(test["Id"]).reset_index()
test_select = test_select.drop(drop,axis=1)
test_select = test_select[test_select.columns & perm_imp_rfpimp]

predictions = pd.DataFrame( rf.predict(test_select),columns=["duration"])
result = pd.merge(test_select_id,predictions,left_on = test_select_id.index,right_on = predictions.index)
result = result.drop(["key_0","index"],axis=1)
result = result.sort_values(by=['Id'])
result["Id"] = result["Id"].astype(int)
result.to_csv("kaggle_submit.txt",sep=",",index=False)

##############################################################



































