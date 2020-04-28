import os 
import numpy as np
import pandas as pd
import codecs, json 
import math

def cal_varn_4 (data_num) :
    
    mat = np.cov(data_num.T)
    mat = np.diag(np.diag(mat))
    return np.divide(mat,100) 


def generate_num_neighbors_4 (inst_num, n, varn):
    return np.random.multivariate_normal(inst_num,varn,n)

def generate_categ_neighbors_4 (inst_categ,n ,mat_nb_categ,special) :
    
     
    rs = np.random.RandomState() 
    p_categ = np.size(inst_categ)
    categ_neigh = np.zeros(n*p_categ).reshape(n,p_categ)
    
    for j in range(0,p_categ) :
        med = int(n/2)

        if j in special :
        	categ_neigh[:med,j] = inst_categ[j] - 1
        else :
        	categ_neigh[:med,j] = inst_categ[j]

        categ_neigh[med:,j] = rs.choice(mat_nb_categ[j], size=(1, med))[0] 
        
    return categ_neigh


def generate_all_neighbors_4(data, num_indices, categ_indices, mat_nb_categ, n_neigh,special) :
    
    list_neigh = []
    n = np.size(data,0)
    data_num = data[:,num_indices]
    varn = cal_varn_4 (data_num)
    
    for i in range(0,n) :
        inst_num = data_num[i]
        num_neigh_i = generate_num_neighbors_4(inst_num,n_neigh,varn)
        
        inst_categ = data[i,categ_indices]
        categ_neigh_i = generate_categ_neighbors_4(inst_categ,n_neigh ,mat_nb_categ,special)
        
        list_neigh.append(np.concatenate((num_neigh_i,categ_neigh_i),axis=1))
        
    return list_neigh





