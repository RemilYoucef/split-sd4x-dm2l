'''
Created on May 25, 2018

@author: anesbendimerad
'''

import os.path

import matplotlib.pyplot as plt
import numpy as np
usedLineWidth=2
import matplotlib as mpl
#mpl.rcParams['hatch.linewidth'] = usedLineWidth  # previous pdf hatch linewidth
mpl.rcParams['hatch.linewidth'] = usedLineWidth 

usedFontSize=23
usedColors=["blue","red","black","green","purple"]
#usedMarkers=["s","o","v","p","<"]
usedMarkerSize=10

unitTimeFactor=1

hatchesDefault=["+","o","-","-."]
    
def display1Bar(x_ticks,data,yLabel,outputFilePath,title="",maxY=-1,logScale=False,hatches=hatchesDefault,colors=usedColors):
    
    xValues=np.arange(len(x_ticks))
    
    plt.figure(1,figsize=(12, 8))
    plt.clf()
    
    bars=plt.bar(xValues,data,  0.4, color='#16A085',label='Frank',linewidth=usedLineWidth)
    
    plt.ylabel(yLabel, fontsize=usedFontSize)
    if logScale:
        plt.yscale("log")
    plt.xticks(range(0, len(x_ticks)), x_ticks,fontsize=usedFontSize)#,rotation=-65)
    plt.tick_params(axis='y', labelsize=usedFontSize)
    plt.xlim(-.8, len(x_ticks)-0.2)
    if maxY!=-1:
        plt.ylim(0, maxY)
    if title!="":
        plt.title(title,fontsize=usedFontSize)
    for cpt in range(len(bars)):
        bars[cpt].set_color(colors[cpt%len(colors)])
        bars[cpt].set_edgecolor("black")
        bars[cpt].set_hatch(hatches[cpt%len(hatches)])
    plt.tight_layout(pad=1.7)
    plt.grid()    
    plt.savefig(outputFilePath)

        
