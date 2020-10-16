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
from sklearn.preprocessing import StandardScaler


from rfpimp import permutation_importances
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

import time
import datetime

train = pd.read_csv("train2.txt")
test = pd.read_csv("test2.txt")

train = train.fillna("")
test = test.fillna("")

full_table = pd.concat([train,test])

def parse_symptoms(df,string):
    if re.search(string,df,re.IGNORECASE):
        val = 1
    else:
        val = 0
    return val

def parse_symptoms_count(df):
    if df.symptoms == "":
        val = 0
    elif re.search(";",df.symptoms):
        val = df.symptoms.split(";")
        val = len(val)
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

def process_date(s):
    if s != "":
        s = time.mktime(datetime.datetime.strptime(s,"%d.%m.%Y").timetuple())
        return s 

def process_age_group(data):
    data = int(data)
    if data >= 0 and 2 >= data:
        val = 1
    elif data >= 3 and 5 >= data:
        val = 2
    elif data >= 6 and 13 >= data:
        val = 3
    elif data >= 14 and 18 >= data:
        val = 4
    elif data >= 19 and 33 >= data:
        val = 5
    elif data >= 34 and 48 >= data:
        val = 6
    elif data >= 49 and 64 >= data:
        val = 7
    elif data >= 65 and 78 >= data:
        val = 8
    elif data >= 78 and 98 >= data:
        val = 9
    else:
        val = 0
    return val

def time_cat(data):
    start = time.mktime(datetime.datetime.strptime("13.02.2020","%d.%m.%Y").timetuple())
    end = time.mktime(datetime.datetime.strptime("14.02.2020","%d.%m.%Y").timetuple())
    if data >= start and data <=end:
        val = 2
    elif data > end:
        val = 1
    else:
        val = 0
    return val

def parse_major(data):
    if data == "":
        val = 0
    elif not re.search("cough",data,re.IGNORECASE) and not re.search("fever",data,re.IGNORECASE):
        val = 2
    else:
        val = 1
    return val
    

##############################################################


def process(full_table):
    # table_city = table_freq(train, "city")
    # table_country = table_freq(train, "country")
    # table_province = table_freq(train, "province")
    # table_v1 = table_freq(train, "V1")
    
    table = table_freq(full_table, "country")
    table = table.sort_values(by=['country_new'],ascending=False)
    table = table.reset_index(drop=True)
    select_country = table[(table.index == 0) | (table.country_new <= 2)]
    # full_table = parse_freq(full_table, "country")
    
    table = table_freq(full_table, "province")
    table = table.sort_values(by=['province_new'],ascending=False)
    table = table.reset_index(drop=True)
    select_province= table[(table.index == 0) | (table.province_new <= 2)]
    # full_table = parse_freq(full_table, "province")
    
    table = table_freq(full_table, "city")
    table = table.sort_values(by=['city_new'],ascending=False)
    table = table.reset_index(drop=True)
    select_city = table[(table.index == 0) | (table.city_new <= 2)]
    # full_table = parse_freq(full_table, "city")
    
    table = table_freq(full_table, "V1")
    table = table.sort_values(by=['V1_new'],ascending=False)
    table = table.reset_index(drop=True)
    select_V1 = table[(table.index == 0) | (table.V1_new <= 2)]
    
    full_table["confirmed"] = full_table["confirmed"].apply(lambda x:process_date(x))
    mean_date = full_table["confirmed"].mean(skipna=True)
    full_table["confirmed"] = full_table["confirmed"].fillna(mean_date)
    full_table["date_grp"] = full_table["confirmed"].apply(lambda x:time_cat(x))
    full_table["date_grp"] = full_table["date_grp"].astype('category')
    full_table["confirmed"] = StandardScaler().fit_transform(full_table[["confirmed"]])
    
    full_table["age_new"] = full_table.apply(parse_age, axis=1)
    mean_age = full_table["age_new"].mean(skipna=True)
    full_table["age_new"] = full_table["age_new"].fillna(mean_age)
    full_table["age_new"] = full_table["age_new"].astype(int)
    
    full_table['maj_symptoms_count'] = full_table.apply(parse_symptoms_count, axis=1)
    
    full_table['throat'] = full_table["symptoms"].apply(lambda x:parse_symptoms(x,"throat"))
    full_table["throat"] = full_table["throat"].astype('category')
    
    # full_table['diarrhea'] = full_table["symptoms"].apply(lambda x:parse_symptoms(x,"diarrhea"))
    # full_table["diarrhea"] = full_table["diarrhea"].astype('category')
    
    full_table['diarr'] = full_table["symptoms"].apply(lambda x:parse_symptoms(x,"diarr"))
    full_table['diarr'] = full_table["diarr"].astype('category')
    
    # full_table['cough'] = full_table["symptoms"].apply(lambda x:parse_symptoms(x,"cough"))
    # full_table["cough"] = full_table["cough"].astype('category')
    
    # full_table['fever'] = full_table["symptoms"].apply(lambda x:parse_symptoms(x,"fever"))
    # full_table["fever"] = full_table["fever"].astype('category')
    
    # full_table['pneu'] = full_table["symptoms"].apply(lambda x:parse_symptoms(x,"pneu"))
    # full_table['pneu'] = full_table["pneu"].astype('category')
    
    # full_table['phary'] = full_table["symptoms"].apply(lambda x:parse_symptoms(x,"phary"))
    # full_table['phary'] = full_table["phary"].astype('category')
    
    # full_table['fati'] = full_table["symptoms"].apply(lambda x:parse_symptoms(x,"fati"))
    # full_table['fati'] = full_table["fati"].astype('category')
    
    full_table['non_maj'] = full_table["symptoms"].apply(lambda x:parse_major(x))
    full_table["non_maj"] = full_table["non_maj"].astype('category')
    
    full_table['agrp'] = full_table["age_new"].apply(lambda x:process_age_group(x))
    
    
    # full_table["city"] = full_table.city + full_table.province
    full_table["city"] = full_table["city"].astype('category')
    # full_table["city"] = full_table["city"].cat.codes
    
    # full_table["province"] = full_table.province + full_table.country
    full_table["province"] = full_table["province"].astype('category')
    # full_table["province"] = full_table["province"].cat.codes
    
    # full_table["V1"] = full_table["V1"].astype('category')
    # full_table["V1"] = full_table["V1"].cat.codes
    
    full_table["country"] = full_table["country"].astype('category')
    # full_table["country"] = full_table["country"].cat.codes
    
    # full_table["outcome"] = full_table["outcome"].astype('category')
    # full_table["outcome"] = full_table["outcome"].cat.codes
    
    # full_table["sex"] = full_table["sex"].astype('category')
    # full_table["sex"] = full_table["sex"].cat.codes
    
    
    full_table = pd.get_dummies(full_table, 
                                      columns= ["city", "province", "country",'V1'], 
                                      prefix = ["city", "province", "country","V1"],drop_first=True)
    
    
    for i in select_city.city:
        full_table = full_table[full_table.columns.drop(list(full_table.filter(regex=i)))]
    
    for i in select_province.province:
        full_table = full_table[full_table.columns.drop(list(full_table.filter(regex=i)))]
        
    for i in select_country.country:
        full_table = full_table[full_table.columns.drop(list(full_table.filter(regex=i)))]
    
    # for i in select_V1.V1:
    #     full_table = full_table[full_table.columns.drop(list(full_table.filter(regex=i)))]
    
    return full_table

# train = full_table[np.isnan(full_table.Id)]
# test = full_table[pd.notnull(full_table.Id)]



train = process(train)
test = process(test)

train_select = train
train_select_y = train["duration"]
drop = ['age','duration','symptoms','sex','outcome']
train_select_x = train_select.drop(drop,axis=1)
train_select_x = train_select_x[train_select_x.columns & test.columns]
train_features, test_features, train_labels, test_labels = train_test_split(train_select_x, train_select_y, test_size = 0.20, random_state = 42)


##############################################################

rf = RandomForestRegressor(n_estimators=200,min_samples_split= 20, 
                           min_samples_leaf= 2, max_features= 'sqrt', 
                           max_depth= 40, bootstrap= True,
                           random_state = 2000)
rf.fit(train_features,train_labels)
print('R^2 Training Score: {:.2f}\nR^2 Validation Score: {:.2f}\n'.format(rf.score(train_features, train_labels),
                                                                                              rf.score(test_features, test_labels)))


# feat_import = rf.feature_importances_

def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))

perm_imp_rfpimp = permutation_importances(rf, train_features, train_labels, r2)
perm_imp_rfpimp.reset_index(level=0, inplace=True)
perm_imp_rfpimp = perm_imp_rfpimp[perm_imp_rfpimp.Importance > 0.001]
perm_imp_rfpimp = perm_imp_rfpimp.Feature
train_select_x = train_select_x[train_select_x.columns & perm_imp_rfpimp]


rf.fit(train_select_x,train_select_y)
print('R^2 Training Score: {:.2f} \n'.format(rf.score(train_select_x, train_select_y)))

# ##############################################################

# Validation
drop = ['age','symptoms','sex']
test_select = test[train.columns & test.columns]
test_select_id = pd.DataFrame(test["Id"]).reset_index()
test_select = test.drop(drop,axis=1)
test_select = test_select[test_select.columns & perm_imp_rfpimp]

predictions = pd.DataFrame( rf.predict(test_select),columns=["duration"])
result = pd.merge(test_select_id,predictions,left_on = test_select_id.index,right_on = predictions.index)
result = result.drop(["key_0","index"],axis=1)
result = result.sort_values(by=['Id'])
result["Id"] = result["Id"].astype(int)
result.to_csv("kaggle_submit_rf.txt",sep=",",index=False)

##############################################################




#############################################################
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

# regr = RandomForestRegressor()
# regr_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, n_iter = 500, cv = 10, verbose=2, random_state=42, n_jobs = -1)
# regr_random.fit(train_features, train_labels)

# print(regr_random.best_params_)
##############################################################






'''
    plt.figure(0)
    plt.scatter(train['duration'],train["confirmed"])
    plt.title("confirmed")
    
    plt.figure(1)
    plt.scatter(train['duration'],train["city"])
    plt.title("city")
    
    plt.figure(2)
    plt.scatter(train['duration'],train["province"])
    plt.title("province")
    
    plt.figure(3)
    plt.scatter(train['duration'],train["country"])
    plt.title("country")
    
    plt.figure(4)
    plt.scatter(train['duration'],train["V1"])
    plt.title("V1")
    
    plt.figure(5)
    plt.scatter(train['duration'],train["age_new"])
    plt.title("age")
    
    plt.figure(6)
    plt.scatter(train['duration'],train["sex"])
    plt.title("sex")
    
    plt.figure(7)
    plt.scatter(train['duration'],train["outcome"])
    plt.title("outcome")
    
    plt.figure(8)
    plt.scatter(train['duration'],train["maj_symptoms_count"])
    plt.title("maj sym count")
    
    plt.figure(9)
    plt.scatter(train['duration'],train["agrp"])
    plt.title("agrp")
    
    plt.figure(10)
    plt.scatter(train['duration'],train["date_grp"])
    plt.title("date_grp")
    
    plt.figure(11)
    plt.scatter(train['duration'],train["non_maj"])
    plt.title("non_maj")
    
    
    plt.figure(12)
    plt.hist(train['duration'],bins=30)
    plt.title("duration")

'''





















