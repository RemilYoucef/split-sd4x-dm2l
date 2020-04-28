'''
Created on Mar 21, 2020

@author: anesbendimerad
'''
from plotFunctions.PlotBars import display1Bar
hatchesSet={"A":"+","B":"o","C":"-","D":"-","E":"-","F":"-","G":"-."}
colorsSet={"A":"#3498db","B":"#9b59b6","C":"#e74c3c","D":"#e74c3c","E":"#e74c3c","F":"#e74c3c","G":"#1abc9c"}


def plotOneScore(case,path):
    resultsPath="FIGURES/rsquared_"+case+".png"
    title=case
    hatches=[]
    colors=[]
    with open(path + "f1score.txt","r") as f:
        labels=[]
        values=[]
        for line in f:
            elements=line.replace("\r","").replace("\n","").split(":")
            elements[0]=elements[0].replace("SD-Split","Split4SDX")
            labels.append(elements[0])
            hatches.append(hatchesSet[elements[0]])
            colors.append(colorsSet[elements[0]])
            values.append(float(elements[1]))
    display1Bar(labels, values, "R squared", resultsPath,title=title,maxY=1,hatches=hatches,colors=colors)
    
