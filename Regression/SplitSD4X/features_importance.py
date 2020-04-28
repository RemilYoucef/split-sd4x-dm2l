
import os.path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def sort_subgroups_support(S,K) :
	S_copy = S.copy()
	l_best_s = []
	for i in range(0,K) :
		inter = 0
		s_best = None 

		for s in S_copy :
			if len(s) > inter :
				inter = len(s)
				s_best = s
		l_best_s.append(s_best)
		S_copy.remove(s_best)
	
	return l_best_s


def plot_explanations (name,W_,l_best_s,att_names,patterns,K):


	if name == 'bike' :
	
		a = list(att_names[0:9])
		b = list(att_names[20:-1])
		att_names = a + b
		att_names = att_names[:10]+ att_names [-2:]
	
	for j in range(0,10) :
		s = l_best_s[j]
		print('the subgroup description is :\n'+ patterns[s])
		fig, ax = plt.subplots(figsize=(11, 8))
		f_importance = np.divide(W_[s].params, W_[s].bse, out=np.zeros_like(W_[s].params), where=W_[s].bse!=0)

		if name == 'bike' :

			d = f_importance [0:9]
			e = f_importance[20:-1]
			f_importance = np.concatenate((d,e),axis = 0)
			f_importance = np.concatenate((f_importance[:10],f_importance[-2:]),axis=0)

		f_importance_abs = np.absolute(f_importance)
		ind =  np.argpartition(f_importance_abs, -K)[-K:]

		f_importance_ = f_importance[ind]
		att_names_ = np.take(att_names, ind)

		f_importance_1 = f_importance_[f_importance_>0]
		att_names_1 = att_names_[f_importance_>0]

		f_importance_2 = f_importance_[f_importance_<0]
		att_names_2 = att_names_[f_importance_<0]


		plt.barh(att_names_1, f_importance_1,color='green')
		plt.barh(att_names_2, f_importance_2,color='red')
		plt.xlabel("Features Importance")
		plt.grid() 
		plt.title(name)
		plt.savefig("FIGURES/Features_Importance/"+"sb"+str(j)+".png", format="png",bbox_inches = 'tight')
		plt.show()