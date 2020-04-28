import os 
import os 
import numpy as np
import pandas as pd
import codecs, json 
import math
from sklearn.tree import DecisionTreeClassifier


def cal_covn(data_num) :
    return np.divide(np.cov(data_num.T),100) 


def generate_num_neighbors (inst_num, n, covn):
    return np.random.multivariate_normal(inst_num,covn,n) 


def generate_all_num_neighbors (data_num, n_neigh) :
     
    list_neigh_num = [] 
    n = np.size(data_num, 0) 
    p = np.size(data_num, 1) 
    covn = cal_covn(data_num)
    
    for i in range (0,n) :
        list_neigh_num.append(generate_num_neighbors(data_num[i], n_neigh, covn))
    
    return list_neigh_num

def classifier_categ (data_num, data_categ) :
    
    decisiontree = DecisionTreeClassifier(splitter='best',random_state=0)
    model = decisiontree.fit(data_num, data_categ)
    return model

def predict_categ (data, data_num, rank_categ, id_inst, list_neigh_num) :
    
    model = classifier_categ (data_num, data[:,rank_categ])
    return model.predict(list_neigh_num[id_inst])


def generate_all_neighbors(data, num_indices, categ_indices, n_neigh):
    
    data_num = data[:,num_indices]
    list_neigh_num = generate_all_num_neighbors (data_num, n_neigh)
    list_neigh = list_neigh_num [:]
    
    n = np.size(data,0)
    
    for j in categ_indices :
        for i in range(0,n) :
            data_categ_j = predict_categ (data, data_num, j, i, list_neigh_num)
            list_neigh[i] = np.insert(list_neigh[i], j, data_categ_j, axis=1)

    return list_neigh


def convert_all_data_to_json (data,target,list_neigh,path = "/data.json") :
    
    n = np.size(data,0)
    list_data = []
    
    for i in range(0,n) :
        list_data.append({"att_values":round_float(data[i,:],3).tolist(), 
                          "target":round_float(target[i],3), 
                          "neighbors":round_float(list_neigh[i],3).tolist()})
    

    file_path = path ## path variable
    json.dump(list_data, codecs.open(file_path, 'w', encoding='utf-8'), 
              separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format



def round_float(data,nb_decimals):
    return np.around(data,decimals = nb_decimals)