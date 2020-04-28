'''
Created on Mar 28, 2020

@author: anesbendimerad
'''

from plotFunctions.PlotCurves import plotOneCase


def plotMSEwrtNbModels(case,path):
    resultsPath="FIGURES/MSE_Gen"+case+".png"
    title=case
    #cpt=0
    #dividor=1
    nbObjects=float(1)
    keys=["Generation_1","Generation_2","Generation_3","Generation_4"]
    legends=["gen-1","gen-2","gen-3","gen-4"]
    yValues=[[],[],[],[]]
    index=-1
    with open(path + "mse_generation.txt","r") as f:
        for line in f:
            if line.replace("\n","").replace("\r","") in keys:
                index+=1
            else:
                yValues[index].append(float(line)/nbObjects)
    xValues=[i+1 for i in range(len(yValues[0]))]    
    plotOneCase(xValues, yValues, "# subgroups (K)", "MSE", legends, resultsPath, title, useMarker=[True,True,True,True],legendsLoc="upper right",markevery=int(15*len(xValues)/100))

