'''
Created on Mar 21, 2020

@author: anesbendimerad
'''
from plotFunctions.PlotCurves import plotOneCase


def plotMSE(case,path):
    resultsPath="FIGURES/MSE_"+case+".png"
    title=case
    #cpt=0
    #dividor=1
    with open(path + "mse.txt","r") as f:
        f.readline()
        globalModel=float(f.readline())
        f.readline()
        localModel=float(f.readline())
        f.readline()
        sdModels=[]        
        for line in f:
            #if cpt%dividor==0:
            sdModels.append(float(line))
            #cpt+=1
    xValues=[i+1 for i in range(len(sdModels))]
    yValues=[[globalModel]*len(sdModels),sdModels,[localModel]*len(sdModels)]    
    plotOneCase(xValues, yValues, "# subgroups (K)", "MSE", ["global-wb","SplitSD4X","local-wb"], resultsPath, title, useMarker=[False,True,False],linestyles=["--","-","-."],legendsLoc="right",markevery=int(15*len(sdModels)/100))


    