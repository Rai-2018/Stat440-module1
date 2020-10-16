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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
import time
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from rfpimp import permutation_importances
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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
    
    # if data >= 0 and 19 >= data:
    #     val = 1
    # elif data >= 20 and 29 >= data:
    #     val = 2
    # elif data >= 30 and 39 >= data:
    #     val = 3
    # elif data >= 40 and 49 >= data:
    #     val = 4
    # elif data >= 50 and 59 >= data:
    #     val = 5
    # elif data >= 60 and 69 >= data:
    #     val = 6
    # elif data >= 70 and 79 >= data:
    #     val = 7
    # else:
    #     val = 8
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
    '''
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
    
    full_table['diarr'] = full_table["symptoms"].apply(lambda x:parse_symptoms(x,"diarr"))
    full_table['diarr'] = full_table["diarr"].astype('category')
    
    full_table = pd.get_dummies(full_table, 
                                      columns= ["city", "province", "country",'V1'], 
                                      prefix = ["city", "province", "country","V1"],drop_first=False)
    return full_table
    ''' 
    def table_freq(df,col):
        name = col+"_new"
        count = df.groupby([col]).size().reset_index(name=name)
        return count

    def parse_freq(df,col):
        return pd.merge(df, table, left_on=col, right_on=col)
    
    table = table_freq(full_table, "country")
    table = table.sort_values(by=['country_new'],ascending=False)
    table = table.reset_index(drop=True)
    select_country = table[(table.index == 0) | (table.country_new < 2)]
    # full_table = parse_freq(full_table, "country")
    
    table = table_freq(full_table, "province")
    table = table.sort_values(by=['province_new'],ascending=False)
    table = table.reset_index(drop=True)
    select_province= table[(table.index == 0) | (table.province_new < 2)]
    # full_table = parse_freq(full_table, "province")
    
    table = table_freq(full_table, "city")
    table = table.sort_values(by=['city_new'],ascending=False)
    table = table.reset_index(drop=True)
    select_city = table[(table.index == 0) | (table.city_new < 2)]
    # full_table = parse_freq(full_table, "city")
    
    table = table_freq(full_table, "V1")
    table = table.sort_values(by=['V1_new'],ascending=False)
    table = table.reset_index(drop=True)
    select_V1 = table[(table.index == 0) | (table.V1_new < 2)]
    
    full_table["confirmed"] = full_table["confirmed"].apply(lambda x:process_date(x))
    mean_date = full_table["confirmed"].mean(skipna=True)
    full_table["confirmed"] = full_table["confirmed"].fillna(mean_date)
    # full_table["date_grp"] = full_table["confirmed"].apply(lambda x:time_cat(x))
    # full_table["date_grp"] = full_table["date_grp"].astype('category')
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
    # full_table["city_cat"] = full_table["city"].cat.codes
    
    # full_table["province"] = full_table.province + full_table.country
    full_table["province"] = full_table["province"].astype('category')
    # full_table["province_cat"] = full_table["province"].cat.codes
    
    full_table["V1"] = full_table["V1"].astype('category')
    # full_table["V1"] = full_table["V1"].cat.codes
    
    full_table["country"] = full_table["country"].astype('category')
    # full_table["country_cat"] = full_table["country"].cat.codes
    
    # full_table["outcome"] = full_table["outcome"].astype('category')
    # full_table["outcome"] = full_table["outcome"].cat.codes
    
    full_table["sex"] = full_table["sex"].astype('category')
    full_table["sex"] = full_table["sex"].cat.codes
    
    
    full_table = pd.get_dummies(full_table, 
                                      columns= ["city", "province", "country",'V1'], 
                                      prefix = ["city", "province", "country","V1"],drop_first=True)
    
    # for i in select_city.city:
    #     full_table = full_table[full_table.columns.drop(list(full_table.filter(regex=i)))]
    
    # for i in select_province.province:
    #     full_table = full_table[full_table.columns.drop(list(full_table.filter(regex=i)))]
        
    # for i in select_country.country:
    #     full_table = full_table[full_table.columns.drop(list(full_table.filter(regex=i)))]
    
    # for i in select_V1.V1:
    #     full_table = full_table[full_table.columns.drop(list(full_table.filter(regex=i)))]
    
    return full_table
train = process(train)
test = process(test)


# keep = ["province_38fc4", "province_38fc4" , "country_59dcd" , "city_0bd76" , "V1_dd554" ]
# keep = train[train.columns &keep]
drop = ['age','sex','symptoms','outcome','duration']

# must_keep = ["confirmed","age_new","maj_symptoms_count","throat"
#              "diarr","non_maj","agrp"]
train_x = train.drop(drop,axis=1)
train_x = train_x[train_x.columns & test.columns ]
# train_x = pd.merge(train_x,keep,left_on = train_x.index,right_on = keep.index)
# train_x = train_x.drop("key_0",axis=1)

train_y = train["duration"]


drop = ['age','sex','symptoms','Id']
test_x = test.drop(drop,axis=1)
test_x = test_x[train_x.columns & test_x.columns]

model_bic = Ridge()
parameters = {'alpha':[1,2,3,4,5,10,
                        11,12,13,14,15,
                        16,17,18,19,
                        20,21,22,23,24,25,
                        26,27,28,29,30,40,
                        50,60,70,80,90],
              }
Ridge_reg = GridSearchCV(model_bic, parameters, scoring='neg_mean_squared_error',cv = 20)
Ridge_reg.fit(train_x, train_y)
print('\n Start',Ridge_reg.best_params_)

model_bic = Ridge(alpha= 22)
model_bic.fit(train_x, train_y)

test_select_id = test['Id']
predictions = pd.DataFrame( model_bic.predict(test_x),columns=["duration"])

result = pd.merge(test_select_id,predictions,left_on = test_select_id.index,right_on = predictions.index)
result = result.drop(["key_0"],axis=1)
result = result.sort_values(by=['Id'])
result["Id"] = result["Id"].astype(int)
result.to_csv("kaggle_submit_lr.txt",sep=",",index=False)


# reg = LinearRegression()
# reg.fit(train_x, train_y)

# prediction = reg.predict(test_x)

# model_bic = Lasso()
# parameters = {'alpha':[1,2,3,4,5,10,
#                        11,12,13,14,15,
#                        16,17,18,19,
#                        20,21,22,23,24,25,
#                        26,27,28,29,30,40,
#                        50,60,70,80,90],
#               "max_iter":[1000,2000,3000,4000],
#               "normalize":[True]
#               }
# Ridge_reg = GridSearchCV(model_bic, parameters, scoring='neg_root_mean_squared_error',cv = 20)
# Ridge_reg.fit(train_x, train_y)
# print(Ridge_reg.best_params_)

# model_bic =  Lasso(alpha=1,max_iter=1000,normalize=(True))
# model_bic.fit(train_x, train_y)
# predict = model_bic.predict(test_x)

# test_select_id = test['Id']
# predictions = predictions = pd.DataFrame( model_bic.predict(test_x),columns=["duration"])
# result = pd.merge(test_select_id,predictions,left_on = test_select_id.index,right_on = predictions.index)
# result = result.drop(["key_0"],axis=1)
# result = result.sort_values(by=['Id'])
# result["Id"] = result["Id"].astype(int)
# result.to_csv("kaggle_submit_lr1.txt",sep=",",index=False)

# model_bic.fit(train_x, train_y)
# predict = model_bic.predict(test_x)



# parameters = {'alpha':[1,2,3,4,5,10,
#                         11,12,13,14,15,
#                         16,17,18,19,
#                         20,21,22,23,24,25,
#                         26,27,28,29,30,40,
#                         50,60,70,80,90],
#               "fit_intercept":[True,False],
#               'solver':["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
#               }


# parameters = {'alpha':[1,2,3,4,5,10,
#                         11,12,13,14,15,
#                         16,17,18,19,
#                         20,21,22,23,24,25,
#                         26,27,28,29,30,40,
#                         50,60,70,80,90,1e-15,1e-10,1e-8,1e-3,1e-2],
#               "normalize":[True,False],
#               "max_iter":[None,100,200,500,1000,2000],
#               "fit_intercept":[True,False]
#               }
