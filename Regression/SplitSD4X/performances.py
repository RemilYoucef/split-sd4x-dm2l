import os 
import sys
import numpy as np
import pandas as pd
import codecs, json 
import math

from subgroups_discovery import *
from neighbors_generation import *
from sklearn.metrics import r2_score



def loss_sd (S,data_test,list_neigh,model) :

	loss = 0 
	for s in S :
		data_neigh_s, target_neigh_s = sampling_sb(data_test,s,list_neigh,model)
		loss += calc_loss(data_neigh_s, target_neigh_s)
	return loss


def loss_global_wb (data_test,n,list_neigh,model) :

	data_neigh_O, target_neigh_O = sampling_sb(data_test,np.arange(n),list_neigh,model)
	global_loss = calc_loss(data_neigh_O, target_neigh_O)
	return global_loss


def loss_local_models (n,list_neigh,model) :

	loss = 0
	for i in range(0,n):
		data_neigh_i = list_neigh[i]
		target_neigh_i = model.predict(data_neigh_i)
		loss += calc_loss(data_neigh_i, target_neigh_i)
	return loss


def unit_vector(vector):
	
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):

	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def similarity (W) :
	
	l = []
	for key, value in W.items():
		temp = [key,value]
		l.append(temp)
	
	distance_matrix = np.zeros(len(l)**2).reshape(len(l),len(l))
	for i in range (0,len(l)) :
		for j in range (i,len(l)) :
			v1 = l[i][1].params
			v2 = l[j][1].params
			distance_matrix[i,j] = round(math.cos(angle_between(v1,v2)),5)
			distance_matrix[j,i] = distance_matrix[i,j]
	return distance_matrix


def avg_non_similar (dist,treshold) :
	
	nb_non_sim = 0 
	nb_sbgrps = np.size(dist,0)
	for i in range (0, nb_sbgrps) :
	    for j in range (i+1, nb_sbgrps) :
	        if dist[i,j] <= treshold :
	            nb_non_sim += 1

	return nb_non_sim / (nb_sbgrps * (nb_sbgrps - 1) / 2)    


def rsquared_global_wb(data_test,n,list_neigh,model) :

	data_neigh_O, target_neigh_O = sampling_sb(data_test,np.arange(n),list_neigh,model)
	data_neigh_O = sm.add_constant(data_neigh_O)
	ols = sm.OLS(target_neigh_O,data_neigh_O).fit()
	return ols.rsquared

def rsquared_local_models (n,list_neigh,model) :

	r_sq  = 0 
	for i in range (0,n) :
		data_neigh_i = list_neigh[i]
		target_neigh_i = model.predict(data_neigh_i)
		regression = LinearRegression()
		reg = regression.fit(data_neigh_i, target_neigh_i)
		new_target = reg.predict(data_neigh_i)
		r2_s = r2_score(target_neigh_i,new_target)
		r_sq = r_sq + r2_s
		del reg
		del regression
	
	return r_sq / n  


def rsquared_splime(S_,data_test,n,list_neigh,model) :

	r_sq = 0 
	for s in S_ :

		data_neigh_s, target_neigh_s = sampling_sb(data_test,s,list_neigh,model)
		regression = LinearRegression()
		reg = regression.fit(data_neigh_s, target_neigh_s)
		new_target = reg.predict(data_neigh_s)
		r2_s = r2_score(target_neigh_s,new_target)
		r_sq = r_sq + r2_s * len(s)
		del regression
		del reg
	
	return r_sq / n


def rsquared_sd (S,W,n) :

	r_sq = 0 
	for s in S :
		#print(W[s].rsquared)
		r_sq = r_sq + W[s].rsquared * len(s)

	return r_sq / n















