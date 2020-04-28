import os
import numpy as np
import pandas as pd
import codecs, json
import math

from subgroups_discovery import *
from neighbors_generation import *
from sklearn.preprocessing import StandardScaler


def sp_lime (list_neigh,model,k) :

    (coef_mat, I) = calc_arg_splime (list_neigh,model,k)
    n = np.size(coef_mat,0)
    v = set ()
    stop = False
    while not stop :
        d = dict ()
        for x in range (0,n) :
            if x not in v :
                d[x] = 0
                ind = np.argsort(coef_mat[x])[-k:]
                for j in ind :
                    d[x] += I[j]
        xk = max(d, key = d.get)
        v.add(xk)
        f = np.argsort(coef_mat[xk])[-k:]
        for j in f :
            I[j] = 0
        if all(I == 0) :
            stop = True
    return v


def calc_arg_splime (list_neigh,model,k) :

    n = len(list_neigh)
    p = np.size(list_neigh[0],1)
    coef_mat = np.zeros(n*p).reshape(n,p) # a modifier cette merde

    for i in range(0,n):
        data_neigh_i = list_neigh[i]
        target_neigh_i = model.predict(data_neigh_i)
        regression = LinearRegression()
        reg = regression.fit(data_neigh_i, target_neigh_i)
        ind = np.argsort(np.absolute(reg.coef_))[-k:]
        for j in ind :
            coef_mat[i,j] = math.fabs(reg.coef_[j])
    I = sum(coef_mat)
    return(coef_mat,I)


def K_means (data,centers) :

    n = np.size(data,0)
    clust = np.zeros(n)

    scaler = StandardScaler()
    data_s = scaler.fit_transform(data)

    for x in range(0,n) :
        l_dist = []
        for i in range(0,len(centers)) :
            c = centers[i]
            d = np.linalg.norm(data_s[x]-data_s[c])
            l_dist.append(d)

        index_min = l_dist.index(min(l_dist))
        clust[x] = centers[index_min]


    clust = clust.astype(int)
    l = group_instances (n, clust, centers)
    return l


def sb_splime (data_test,list_neigh,model) :
    v = sp_lime (list_neigh,model,5)
    centers = list(v)
    l = K_means(data_test,centers)
    S_ = set(l)
    return S_


def group_instances (n, inst_clust, centers) :

    l_ = []
    for c in centers :
        l_.append(tuple(np.arange(n)[inst_clust == c]))

    return l_



def get_description_splime (data_test,att_names,S_) :
    p = np.size(data_test,1)
    print('----------------------------')
    for a in range(0,p) :
        print(att_names[a])
        print('----------------------------')
        i = 0
        for s_ in S_  :
            print('Subrgoup',i)
            print('min',np.min(data_test[s_,a]))
            print('min',np.max(data_test[s_,a]))
            i += 1
            print('----------------------------')


def overlap (data_test,s1,s2) :

    logic = []
    p = np.size(data_test,1)
    for a in range(0,p) :
        min_s1 = np.min(data_test[s1,a])
        min_s2 = np.min(data_test[s2,a])
        max_s1 = np.max(data_test[s1,a])
        max_s2 = np.max(data_test[s2,a])

        cond1 = (min_s1 > min_s2 and min_s1 < max_s2)
        cond2 = (min_s2 > min_s1 and min_s2 < max_s1)
        cond3 = (min_s1 == min_s2 or max_s1 == max_s2)

        if cond1 or cond2 or cond3 :
            logic.append(True)
        else :
            logic.append(False)

    return logic


def matrix_overlaps(S_,data_test) :

    n_ = len(S_)
    mat_overlap = np.zeros(n_**2).reshape(n_,n_)
    i = 0
    while len(S_) > 0 :
        s_ = S_.pop()
        if len(S_) > 0 :
            j = i + 1
            for s in S_ :
                if all(elem == True for elem in overlap (data_test,s_,s)) :
                    mat_overlap[i,j] = -1
                    mat_overlap[j,i] = -1
                else :
                    mat_overlap[i,j] = 1
                    mat_overlap[j,i] = 1
                j += 1
        i += 1
    return mat_overlap


def avg_overlaps (mat_overlap) :

    nb_overlaps = 0
    nb_sbgrps = np.size(mat_overlap,0)
    for i in range (0, nb_sbgrps) :
        for j in range (i+1, nb_sbgrps) :
            if mat_overlap[i,j] == -1 :
                nb_overlaps += 1

    return nb_overlaps / (nb_sbgrps * (nb_sbgrps - 1) / 2)


def is_included (instance, data_test, s) :

    p = np.size(data_test,1)
    a = 0
    included = True
    while included and (a < p) :
        min_s = np.min(data_test[s,a])
        max_s = np.max(data_test[s,a])
        if data_test[instance,a] > max_s or data_test[instance,a] < min_s :
            included = False
        a += 1

    return included



# Avg of instances covered by more than one subgroup :

def cover_more (S_,data_test,n) :

    nb_inst_cov = 0
    for i in range(0,n) :
        nb_sb_cov = 0
        for s in S_ :
            if is_included (i, data_test, s) :
                nb_sb_cov += 1
        if nb_sb_cov > 1 :
            nb_inst_cov += 1

    return nb_inst_cov / n


# Precision of subgroups :

def sb_precision (S_,data_test,n) :

    avg = 0
    for s in S_ :
        nb_inst_cov = 0
        for i in range (0,n) :
            if is_included (i, data_test, s) :
                nb_inst_cov += 1
        #print(round(len(s)/nb_inst_cov,2))
        avg += len(s)/nb_inst_cov

    return round(avg/len(S_),2)
