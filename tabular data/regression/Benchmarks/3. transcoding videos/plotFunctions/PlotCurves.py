'''
Created on May 25, 2018

@author: anesbendimerad
'''

import os.path

import matplotlib.pyplot as plt

usedFontSize=30
usedColors=["blue","red","green","black","purple"]
usedMarkers=["o","v","P","*","s","<"]
usedMarkerSize=20
usedLineWidth=4
unitTimeFactor=1
tIndexSias="miningTimeInMS"
tIndexPnr="time (sec)"
plt.rc('font', size=usedFontSize)

def plotOneCase(xValues,yValues,xLabel,yLabel,legends,resultsPath,title="",linestyles=[],logScale=False,xLimit=[],yLimit=[],legendsLoc="lower right",doXLim=True,doYLim=True,useMarker=[],markevery=1):
    plt.figure(1,figsize=(12, 9))
    plt.clf()
    if doXLim:
        if xLimit!=[]:
            plt.xlim(xLimit)
        else:
            plt.xlim([0, xValues[len(xValues) - 1] * 1.1])
    if doYLim:
        if yLimit!=[]:
            plt.ylim(yLimit)
        else:
            plt.ylim([0, max([max(yVal) for yVal in yValues])*1.2])
    if logScale:
        plt.yscale('log')
    for cpt in range(len(yValues)):
        if len(linestyles)==0:
            lst="-"
        else:
            lst=linestyles[cpt]
        if len(useMarker)>0 and useMarker[cpt]:
            plt.plot(xValues[:len(yValues[cpt])], yValues[cpt], color=usedColors[cpt % len(usedColors)], linewidth=usedLineWidth, marker=usedMarkers[cpt % len(usedMarkers)], markersize=usedMarkerSize,markevery=markevery,linestyle=lst)
        else:
            plt.plot(xValues[:len(yValues[cpt])], yValues[cpt], color=usedColors[cpt % len(usedColors)], linewidth=usedLineWidth,markevery=markevery,linestyle=lst)        
    plt.legend(legends, loc=legendsLoc, fontsize=usedFontSize,framealpha=0.7)
    plt.xlabel(xLabel, fontsize=usedFontSize)
    plt.ylabel(yLabel, fontsize=usedFontSize)
    plt.tick_params(axis='x', labelsize=usedFontSize)
    plt.tick_params(axis='y', labelsize=usedFontSize)
    if title!="":
        plt.title(title,fontsize=usedFontSize)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    
    plt.gcf().tight_layout()
    plt.grid()
    plt.savefig(resultsPath)
    plt.show()
    
    
    