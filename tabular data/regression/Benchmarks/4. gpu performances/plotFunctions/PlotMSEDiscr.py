'''
Created on Mar 28, 2020

@author: anesbendimerad
'''

from plotFunctions.PlotCurves import plotOneCase


def plotMSEDiscFreq(case,path):
    resultsPath="FIGURES/MSE_Disc_Freq"+case+".png"
    title=case
    #cpt=0
    #dividor=1
    nbObjects=float(1)
    keys=["Generation_1","Generation_2","Generation_3","Generation_4","Generation_5","Generation_6","Generation_7"]
    legends=["k = 4","k = 5","k = 6","k = 7","k = 8","k = 9","k = 10"]
    yValues=[[],[],[],[],[],[],[]]
    index=-1
    with open(path + "mse_disc_freq.txt","r") as f:
        for line in f:
            if line.replace("\n","").replace("\r","") in keys:
                index+=1
            else:
                yValues[index].append(float(line)/nbObjects)
    xValues=[i+1 for i in range(len(yValues[0]))]    
    plotOneCase(xValues, yValues, "# subgroups (K)", "MSE", legends, resultsPath, title, useMarker=[True,True,True,True,True,True,True],legendsLoc="upper right",markevery=int(15*len(xValues)/100))


def plotMSEDiscWidth(case,path):
    resultsPath="FIGURES/MSE_Disc_Width"+case+".png"
    title=case
    #cpt=0
    #dividor=1
    nbObjects=float(1)
    keys=["Generation_1","Generation_2","Generation_3","Generation_4","Generation_5","Generation_6","Generation_7"]
    legends=["k = 4","k = 5","k = 6","k = 7","k = 8","k = 9","k = 10"]
    yValues=[[],[],[],[],[],[],[]]
    index=-1
    with open(path + "MSE_Disc_Width.txt","r") as f:
        for line in f:
            if line.replace("\n","").replace("\r","") in keys:
                index+=1
            else:
                yValues[index].append(float(line)/nbObjects)
    xValues=[i+1 for i in range(len(yValues[0]))]    
    plotOneCase(xValues, yValues, "# subgroups (K)", "MSE", legends, resultsPath, title, useMarker=[True,True,True,True,True,True,True],legendsLoc="upper right",markevery=int(15*len(xValues)/100))