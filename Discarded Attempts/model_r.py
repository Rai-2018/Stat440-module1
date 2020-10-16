# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:41:12 2020

@author: wlian
"""


import pandas as pd
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time
import datetime
from sklearn.preprocessing import StandardScaler


train = pd.read_csv("train2.txt")
test = pd.read_csv("test2.txt")

train = train.fillna("")
test = test.fillna("")



# full_table = pd.concat([train,test])


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
    start = time.mktime(datetime.datetime.strptime("11.02.2020","%d.%m.%Y").timetuple())
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



# table_city = table_freq(train, "city")
# table_country = table_freq(train, "country")
# table_province = table_freq(train, "province")
# table_v1 = table_freq(train, "V1")

def process(full_table):
    full_table["confirmed"] = full_table["confirmed"].apply(lambda x:process_date(x))
    mean_date = full_table["confirmed"].mean(skipna=True)
    full_table["confirmed"] = full_table["confirmed"].fillna(mean_date)
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
    
    # full_table['pneu'] = full_table["pneu"].astype('category')
    # full_table['pneu'] = full_table["symptoms"].apply(lambda x:parse_symptoms(x,"pneu"))
    
    full_table['phary'] = full_table["symptoms"].apply(lambda x:parse_symptoms(x,"phary"))
    full_table['phary'] = full_table["phary"].astype('category')
    
    # full_table['fati'] = full_table["symptoms"].apply(lambda x:parse_symptoms(x,"fati"))
    # full_table['fati'] = full_table["fati"].astype('category')
    
    full_table['non_maj'] = full_table["symptoms"].apply(lambda x:parse_major(x))
    full_table["non_maj"] = full_table["non_maj"].astype('category')
    
    full_table['agrp'] = full_table["age_new"].apply(lambda x:process_age_group(x))
    
    # full_table["city"] = full_table.city + full_table.province
    # full_table["province"] = full_table.province + full_table.country
    
    full_table["city"] = full_table["city"].astype('category')
    # full_table["city"] = full_table["city"].cat.codes
    
    full_table["province"] = full_table["province"].astype('category')
    # full_table["province"] = full_table["province"].cat.codes
    
    # full_table["V1"] = full_table["V1"].astype('category')
    # full_table["V1"] = full_table["V1"].cat.codes
    
    full_table["date_grp"] = full_table["confirmed"].apply(lambda x:time_cat(x))
    full_table["date_grp"] = full_table["date_grp"].astype('category')
    
    full_table["country"] = full_table["country"].astype('category')
    # full_table["country"] = full_table["country"].cat.codes
    
    # full_table["outcome"] = full_table["outcome"].astype('category')
    # full_table["outcome"] = full_table["outcome"].cat.codes
    
    # full_table["sex"] = full_table["sex"].cat.codes
    # full_table["sex"] = full_table["sex"].astype('category')
    
    full_table = pd.get_dummies(full_table, 
                                      columns= ["city", "province", "country",'V1'], 
                                      prefix = ["city", "province", "country","V1"],drop_first=True)
    
    # full_table_onehot = pd.get_dummies(full_table, 
    #                                   columns= ["city", "province", "country", "V1","outcome"], 
    #                                   prefix = ["city", "province", "country", "V1","outcome"],drop_first=True)
    return full_table

# train = full_table[np.isnan(full_table.Id)]
# test = full_table[pd.notnull(full_table.Id)]

train = process(train)

keep = ["confirmed","age_new","maj_symptoms_count","throat",
        "diarr","province_38fc4","province_55fe6","country_59dcd",
        "city_0bd76","V1_dd554", "V1_9a45a", "V1_f9037"] 

# keep = ["city_d7cac","city_dbe0f",
#         "country_3f760","city_6bcaa", 
#         "province_61aef", "city_b1224",
#         "city_76704","city_ba1b5"] 

train_select = train
train_select_y = train["duration"]
train_select_x = train_select[train_select.columns & keep]

##############################################################

reg =  LinearRegression().fit(train_select_x, train_select_y)
test_select = test
test_select_id = pd.DataFrame(test["Id"]).reset_index()
test_select_x = test_select[test_select.columns & keep]

test = process(test)

predictions = pd.DataFrame( reg.predict(test_select_x),columns=["duration"])
result = pd.merge(test_select_id,predictions,left_on = test_select_id.index,right_on = predictions.index)
result = result.drop(["key_0","index"],axis=1)
result = result.sort_values(by=['Id'])
result["Id"] = result["Id"].astype(int)
result.to_csv("kaggle_submit_lr.txt",sep=",",index=False)



train_features, test_features, train_labels, test_labels = train_test_split(train_select_x, train_select_y, test_size = 0.20, random_state = 42)
print(reg.score(train_features,train_labels))

























