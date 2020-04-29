import os 
import numpy as np
import pandas as pd
import codecs, json 
import math
import time

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from neighbors_generation import *


def SplitBasedSelectionForm_freq (data, target, k, model,list_neigh,split_point,k_disc)  :
	
	# data D = (O,A)
	# b(O) = {b(o1),...,b(o1)}  
	# k a threshold on the number of subgroups
	
	start_time = time.time()
	l_s = []

	n = np.size(data,0) # number of instances
	p = np.size(data,1) # number of features

	regression = LinearRegression() # we are going to use the linear regression
	
	# we define the subgroups by the id of the instances in the data
	Subgroups = set() # S is a set of the subgroups  
	Subgroups.add(tuple(np.arange(n))) # first S is simply all the objects O i.e S = {0}
	
	data_neigh_O, target_neigh_O = sampling_sb(data,np.arange(n),list_neigh,model)
	
	
	patterns = dict()
	# patterns = {attribute : a, value : v, operator : '>' or '<='}
	
	patterns[tuple(np.arange(n))] = (None,None,None)

	
	improv = True
	splits = set ()
	newSubgroups = set()
	newSubgroups.add(tuple(np.arange(n))) # newSubgroups = {O}
	
	loss_subgroups = dict () # for the losses of the subgroups without spliting
	loss_subgroups [tuple(np.arange(n))] = calc_loss(data_neigh_O,target_neigh_O)
	#print('loss_all = ','{:.2e}'.format(loss_subgroups [tuple(np.arange(n))]))
	#print("--- %s seconds ---" % (time.time() - start_time))
	
	
	iteration = 0
	
	while len(Subgroups) < k and improv :
		
		#computing the best splits for the new subgroups : 
		for s in newSubgroups : # s is tuple
			
			if len(s) > 1 : 
			
				list_loss_attributes = []           
				for a in range(0,p) : # iterate over the attributes (i is the attribute)
					
					#print('a =',a)
					min_v = np.min(data[s,a]) #  
					max_v = np.max(data[s,a]) # 
					list_loss_values = []
					
					if min_v != max_v :
						
						if a < split_point :


							# numerical features
							unique_values = np.unique(data[:,a]).size
							#print(unique_values)  

							if unique_values <= 10 :

								unique_values = np.unique(data[s,a]).size

								steps = (pd.cut(data[s,a],unique_values, retbins=True,include_lowest=True))[1][1:-1]
								steps = np.trunc(steps)

							else :
								steps = (pd.cut(data[s,a],k_disc, retbins=True,include_lowest=True))[1][1:-1]
							
						else :
							# categorical features
							steps = (pd.cut(data[s,a],2, retbins=True,include_lowest=True))[1][1:-1]
							
						len_steps = np.size(steps)
						j = 0 
						
						while j < len_steps :
							
							value = steps[j]
							#print('v =',value)

							# split the subgroups :

							# subgroup1 that satisfies the condition s [a > v]
							subgrp1 = tuple(np.asarray(s)[data[s,:][:,a] > value])

							# generating the new dataset of neighbors of the subgroup_1 elements 
							data_neigh_sb1, target_neigh_sb1 = sampling_sb(data,subgrp1,list_neigh,model)

							# subgroup2 that satisfies the condition s [a <= v]
							subgrp2 = tuple(np.asarray(s)[data[s,:][:,a] <= value])

							# generating the new dataset of neighbors of the subgroup_1 elements 
							data_neigh_sb2, target_neigh_sb2 = sampling_sb(data,subgrp2,list_neigh,model)

							#compute the loss and update the loss_subgroups dictionnary

							loss_subgroups[subgrp1] = calc_loss(data_neigh_sb1, target_neigh_sb1)
							loss_subgroups[subgrp2] = calc_loss(data_neigh_sb2, target_neigh_sb2)

							loss =  loss_subgroups[subgrp1] + loss_subgroups[subgrp2]
							#print("loss des 2 sbgrps =",loss)

							# store the losses 
							list_loss_values.append((loss,value))

							#iterate over the j
							j += 1 

					# select the minimum loss and value that minimize the loss for each attribute a 
					if list_loss_values :

						loss_opt_att = min(list_loss_values)

						# store the optimal loss for the attribute 
						list_loss_attributes.append(loss_opt_att)
					
					else :
						list_loss_attributes.append((math.inf,None))



				# select the minimum loss and value that minimize the loss for the subgroup s
				loss_opt_s, value_opt = min(list_loss_attributes)
				attribute_opt = list_loss_attributes.index(min(list_loss_attributes))

				# add the best split for the subgroup (s) to the splits set 
				splits.add((s,attribute_opt,value_opt,loss_opt_s))
		
		# Choose the subgroup split that leads to the minimum loss:
		best_split =  splits.pop()
		tmp_split = best_split # to add it after 
		
		s, a, v, loss_sb = best_split
		
		Subgroups.remove(s)
		best_loss_s = loss_set(Subgroups,loss_subgroups) + loss_sb
		Subgroups.add(s)
		
		
		for split in splits :
			s_, a_, v_, loss_sb_ = split
			Subgroups.remove(s_)
			if loss_set(Subgroups,loss_subgroups) + loss_sb_ < best_loss_s :
				best_loss_s = loss_set(Subgroups,loss_subgroups) + loss_sb_
				best_split = split
			Subgroups.add(s_)
			
		splits.add(tmp_split)
		
		s_best, a_best, v_best, loss_sb_min = best_split
		

		
		if loss_sb_min < loss_subgroups[s_best] :
						
			Subgroups.remove(s_best)
			
			sb1 = tuple(np.asarray(s_best)[data[s_best,:][:,a_best] > v_best])

			
			data_neigh_sb1, target_neigh_sb1 = sampling_sb(data,sb1,list_neigh,model)
			
			sb2 = tuple(np.asarray(s_best)[data[s_best,:][:,a_best] <= v_best])
			data_neigh_sb2, target_neigh_sb2 = sampling_sb(data,sb2,list_neigh,model)

			Subgroups.add(sb1)
			Subgroups.add(sb2)
			
			if iteration == 0 :
				del patterns[s_best]
				patterns[sb1] = (a_best,'>',v_best)
				patterns[sb2] = (a_best,'<=',v_best)
				
			else : 
				
				patterns[sb1] = patterns[s_best] + (a_best,'>',v_best)
				patterns[sb2] = patterns[s_best] + (a_best,'<=',v_best)
				del patterns[s_best]
			
			newSubgroups = {sb1, sb2}
			splits.remove(best_split)
		else :
			improv = False
		
		iteration = iteration + 1
		#print('{:.2e}'.format(loss_set(Subgroups,loss_subgroups)))
		#print("--- %s seconds ---" % (time.time() - start_time))


		S_copy = set ()
		S_copy = Subgroups.copy()

		l_s.append(S_copy)
			
	return(l_s,patterns)


def SplitBasedSelectionForm_width (data, target, k, model,list_neigh,split_point,k_disc)  :
	
	# data D = (O,A)
	# b(O) = {b(o1),...,b(o1)}  
	# k a threshold on the number of subgroups
	
	start_time = time.time()
	l_s = []

	n = np.size(data,0) # number of instances
	p = np.size(data,1) # number of features

	regression = LinearRegression() # we are going to use the linear regression
	
	# we define the subgroups by the id of the instances in the data
	Subgroups = set() # S is a set of the subgroups  
	Subgroups.add(tuple(np.arange(n))) # first S is simply all the objects O i.e S = {0}
	
	data_neigh_O, target_neigh_O = sampling_sb(data,np.arange(n),list_neigh,model)
	
	
	patterns = dict()
	# patterns = {attribute : a, value : v, operator : '>' or '<='}
	
	patterns[tuple(np.arange(n))] = (None,None,None)

	
	improv = True
	splits = set ()
	newSubgroups = set()
	newSubgroups.add(tuple(np.arange(n))) # newSubgroups = {O}
	
	loss_subgroups = dict () # for the losses of the subgroups without spliting
	loss_subgroups [tuple(np.arange(n))] = calc_loss(data_neigh_O,target_neigh_O)
	#print('loss_all = ','{:.2e}'.format(loss_subgroups [tuple(np.arange(n))]))
	#print("--- %s seconds ---" % (time.time() - start_time))
	
	
	iteration = 0
	
	while len(Subgroups) < k and improv :
		
		#computing the best splits for the new subgroups : 
		for s in newSubgroups : # s is tuple
			
			if len(s) > 1 : 
			
				list_loss_attributes = []           
				for a in range(0,p) : # iterate over the attributes (i is the attribute)
					
					#print('a =',a)
					min_v = np.min(data[s,a]) #  
					max_v = np.max(data[s,a]) # 
					list_loss_values = []
					
					if min_v != max_v :
						
						if a < split_point :


							# numerical features
							unique_values = np.unique(data[:,a]).size
							#print(unique_values)  

							if unique_values <= 10 :

								unique_values = np.unique(data[s,a]).size

								steps = np.linspace(min_v, max_v, num =unique_values)[1:-1]
								steps = np.trunc(steps)

							else :
								steps = np.linspace(min_v, max_v, num = k_disc)[1:-1]
							
						else :
							# categorical features
							steps = (pd.cut(data[s,a],2, retbins=True,include_lowest=True))[1][1:-1]
							
						len_steps = np.size(steps)
						j = 0 
						
						while j < len_steps :
							
							value = steps[j]
							#print('v =',value)

							# split the subgroups :

							# subgroup1 that satisfies the condition s [a > v]
							subgrp1 = tuple(np.asarray(s)[data[s,:][:,a] > value])

							# generating the new dataset of neighbors of the subgroup_1 elements 
							data_neigh_sb1, target_neigh_sb1 = sampling_sb(data,subgrp1,list_neigh,model)

							# subgroup2 that satisfies the condition s [a <= v]
							subgrp2 = tuple(np.asarray(s)[data[s,:][:,a] <= value])

							# generating the new dataset of neighbors of the subgroup_1 elements 
							data_neigh_sb2, target_neigh_sb2 = sampling_sb(data,subgrp2,list_neigh,model)

							#compute the loss and update the loss_subgroups dictionnary

							loss_subgroups[subgrp1] = calc_loss(data_neigh_sb1, target_neigh_sb1)
							loss_subgroups[subgrp2] = calc_loss(data_neigh_sb2, target_neigh_sb2)

							loss =  loss_subgroups[subgrp1] + loss_subgroups[subgrp2]
							#print("loss des 2 sbgrps =",loss)

							# store the losses 
							list_loss_values.append((loss,value))

							#iterate over the j
							j += 1 

					# select the minimum loss and value that minimize the loss for each attribute a 
					if list_loss_values :

						loss_opt_att = min(list_loss_values)

						# store the optimal loss for the attribute 
						list_loss_attributes.append(loss_opt_att)
					
					else :
						list_loss_attributes.append((math.inf,None))



				# select the minimum loss and value that minimize the loss for the subgroup s
				loss_opt_s, value_opt = min(list_loss_attributes)
				attribute_opt = list_loss_attributes.index(min(list_loss_attributes))

				# add the best split for the subgroup (s) to the splits set 
				splits.add((s,attribute_opt,value_opt,loss_opt_s))
		
		# Choose the subgroup split that leads to the minimum loss:
		best_split =  splits.pop()
		tmp_split = best_split # to add it after 
		
		s, a, v, loss_sb = best_split
		
		Subgroups.remove(s)
		best_loss_s = loss_set(Subgroups,loss_subgroups) + loss_sb
		Subgroups.add(s)
		
		
		for split in splits :
			s_, a_, v_, loss_sb_ = split
			Subgroups.remove(s_)
			if loss_set(Subgroups,loss_subgroups) + loss_sb_ < best_loss_s :
				best_loss_s = loss_set(Subgroups,loss_subgroups) + loss_sb_
				best_split = split
			Subgroups.add(s_)
			
		splits.add(tmp_split)
		
		s_best, a_best, v_best, loss_sb_min = best_split
		

		
		if loss_sb_min < loss_subgroups[s_best] :
						
			Subgroups.remove(s_best)
			
			sb1 = tuple(np.asarray(s_best)[data[s_best,:][:,a_best] > v_best])

			
			data_neigh_sb1, target_neigh_sb1 = sampling_sb(data,sb1,list_neigh,model)
			
			sb2 = tuple(np.asarray(s_best)[data[s_best,:][:,a_best] <= v_best])
			data_neigh_sb2, target_neigh_sb2 = sampling_sb(data,sb2,list_neigh,model)

			Subgroups.add(sb1)
			Subgroups.add(sb2)
			
			if iteration == 0 :
				del patterns[s_best]
				patterns[sb1] = (a_best,'>',v_best)
				patterns[sb2] = (a_best,'<=',v_best)
				
			else : 
				
				patterns[sb1] = patterns[s_best] + (a_best,'>',v_best)
				patterns[sb2] = patterns[s_best] + (a_best,'<=',v_best)
				del patterns[s_best]
			
			newSubgroups = {sb1, sb2}
			splits.remove(best_split)
		else :
			improv = False
		
		iteration = iteration + 1
		#print('{:.2e}'.format(loss_set(Subgroups,loss_subgroups)))
		#print("--- %s seconds ---" % (time.time() - start_time))


		S_copy = set ()
		S_copy = Subgroups.copy()

		l_s.append(S_copy)
			
	return(l_s,patterns)



