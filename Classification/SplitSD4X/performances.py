import os 
import numpy as np
import pandas as pd
import codecs, json 
import math
from sklearn.metrics import f1_score

from subgroups_discovery import *
from neighbors_generation import *


def loss_sd (S,data_test,list_neigh,model) :

	loss = 0 
	for s in S :
		data_neigh_s, target_neigh_s = sampling_sb(data_test,s,list_neigh,model)
		target_neigh_s_proba = model.predict_proba(data_neigh_s)
		loss += calc_loss(data_neigh_s,target_neigh_s_proba)

	return loss


def loss_global_wb (data_test,list_neigh,model) :

	n = np.size(data_test,0)
	data_neigh_O, target_neigh_O = sampling_sb(data_test,np.arange(n),list_neigh,model)
	target_neigh_O_proba = model.predict_proba(data_neigh_O)
	global_loss = calc_loss(data_neigh_O, target_neigh_O_proba)
	return global_loss


def loss_local_models (n,list_neigh,model) :

	loss = 0
	for i in range(0,n) :
		data_neigh_i = list_neigh[i]
		target_neigh_i = model.predict(data_neigh_i)
		target_neigh_i_proba = model.predict_proba(data_neigh_i)
		loss += calc_loss(data_neigh_i, target_neigh_i_proba)
	
	return loss

'----------------------------------------------------------------------------------------------------'

def unit_vector(vector):
	
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):

	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def similarity (W,nb_classes) :

	l = []

	for key, value in W.items():
		temp = [key,value]
		l.append(temp)


	distance_matrix = np.zeros(len(l)**2).reshape(len(l),len(l))
	for i in range (0,len(l)) :
		for j in range (i,len(l)):
			for c in range (0,nb_classes) :
				if c == 0 : 
					v1 = l[i][1][c].params
					v2 = l[j][1][c].params
				else :
					v1 = np.concatenate((v1,l[i][1][c].params),axis=0)
					v2 = np.concatenate((v2,l[j][1][c].params),axis=0)                    
			distance_matrix[i,j] = round(math.cos(angle_between(v1,v2)),2)
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


def fscore_global_wb (data_test,n,list_neigh,model,nb_classes) :
	
	data_neigh_O, target_neigh_O = sampling_sb(data_test,np.arange(n),list_neigh,model)
	target_neigh_O_proba = model.predict_proba(data_neigh_O)
	lr = LinearRegression() 
	model_lr = lr.fit(data_neigh_O,target_neigh_O_proba)
	target_lr = model_lr.predict(data_neigh_O)
	a = np.argmax(target_lr, axis=1) 
	b = np.argmax(target_neigh_O_proba, axis=1)

	if nb_classes == 2 : 
	
		return f1_score(a,b)

	else :
		return f1_score(a,b,average='macro')


def fscore_sd (S,data_test,list_neigh,model,nb_classes) :

	iteration = 0 
	for s in S :
		data_neigh_s, target_neigh_s = sampling_sb(data_test,s,list_neigh,model)
		target_neigh_s_proba = model.predict_proba(data_neigh_s)
		lr = LinearRegression()
		model_lr = lr.fit(data_neigh_s,target_neigh_s_proba)
		target_lr = model_lr.predict(data_neigh_s)
		if iteration == 0 :
			a = np.argmax(target_lr, axis=1) 
			b = np.argmax(target_neigh_s_proba, axis=1)

		else :
			a = np.concatenate((a,np.argmax(target_lr, axis=1)))
			b = np.concatenate((b,np.argmax(target_neigh_s_proba, axis=1)))

		iteration += 1
		
	if nb_classes == 2 : 
	
		return f1_score(a,b)

	else :
		return f1_score(a,b,average='macro')



def fscore_local_models (data_test,n,list_neigh,model,nb_classes) :
	

	iteration = 0 
	for i in range(0,n) :
		data_neigh_i = list_neigh[i]
		target_neigh_i_proba = model.predict_proba(data_neigh_i)
		lr = LinearRegression()
		model_lr = lr.fit(data_neigh_i,target_neigh_i_proba)
		target_lr = model_lr.predict(data_neigh_i)
		if iteration == 0 :
			a = np.argmax(target_lr, axis=1) 
			b = np.argmax(target_neigh_i_proba, axis=1)

		else :
			a = np.concatenate((a,np.argmax(target_lr, axis=1)))
			b = np.concatenate((b,np.argmax(target_neigh_i_proba, axis=1)))

		iteration += 1
	
	if nb_classes == 2 : 
	
		return f1_score(a,b)

	else :
		return f1_score(a,b,average='macro')











