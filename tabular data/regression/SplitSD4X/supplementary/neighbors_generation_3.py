import os 
import numpy as np
import pandas as pd
import codecs, json 
import math

def cal_varn_3 (data_num) :
    
    mat = np.cov(data_num.T)
    return np.divide(mat,100) 


def generate_num_neighbors_3 (inst_num, n, varn):
    return np.random.multivariate_normal(inst_num,varn,n)

def generate_categ_neighbors_3 (inst_categ,n ,mat_nb_categ) :
    
     
    p_categ = np.size(inst_categ)
    categ_neigh = np.zeros(n*p_categ).reshape(n,p_categ)
    
    for j in range(0,p_categ) :
        rs = np.random.RandomState(11111)
        categ_neigh[:,j] = rs.choice(mat_nb_categ[j], size=(1, n))[0]  
        
    return categ_neigh


def generate_all_neighbors_3(data, num_indices, categ_indices, mat_nb_categ, n_neigh) :
    
    list_neigh = []
    n = np.size(data,0)
    data_num = data[:,num_indices]
    varn = cal_varn_3 (data_num)
    
    for i in range(0,n) :
        inst_num = data_num[i]
        num_neigh_i = generate_num_neighbors_3(inst_num,n_neigh,varn)
        
        inst_categ = data[i,categ_indices]
        categ_neigh_i = generate_categ_neighbors_3(inst_categ,n_neigh ,mat_nb_categ)
        
        list_neigh.append(np.concatenate((num_neigh_i,categ_neigh_i),axis=1))
        
    return list_neigh