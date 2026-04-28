# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 03:48:28 2025

@author: spenc
"""
import math
import numpy as np;
import matplotlib.pyplot as plt;
#delta T is time elapsed since pumping began
#injection time
t0 = 36000; #10 hours

hourTime = 3600;
dayTime = 86400 #24 hours
weekTime = 604800 #1week in sec
yearTime =3.154e+7;
def getDelta(deltaT):
    return deltaT/t0; 

def getg(d):
    return (4/3) * ((1+d)**(3/2) - d**(3/2) -1);

def getG(df,d0):
    return (4/math.pi) * (getg(df) -getg(d0));

def getC(deltaP):
    k = 10**(-18); #permeability
    c = 4.6*10**(-10); #compressibility
    phi = .05; #porosity
    mu = .001006;
    return ((k*c*phi)/(math.pi * mu))**(1/2)*deltaP;

def get_eBar(poisson,nu):
    return poisson/(1-nu**2)
    
#delta = deltaT/t0
def deltaP(timeFinal,timeInitial,CVal):
    Hr = .3;
    H = 100;
    beta = 1;
    deltaF = getDelta(timeFinal);
    deltaI = getDelta(timeInitial)
    C = CVal;
    G = getG(deltaI,deltaF)
    eBar = get_eBar(3*10**10, .35)
    return (C*Hr*eBar * (t0)**(1/2))/(H) * G;


def interpolateGraph():
    initialPressure =  58236090.90178592;
    resPressure= 2 * 10**7;
    initialDeltaP = initialPressure -resPressure;
    initialC = getC(initialDeltaP);
    sampleNum = 500;
    #finalTime = 10*weekTime;
    finalTime = yearTime;
    times = np.linspace(3600,yearTime, num = sampleNum);
    output = np.zeros(sampleNum);
    priorC = initialC
    priorP = initialDeltaP;
    for i in range(len(output)-1):
        dP = deltaP(times[i+1],times[i],priorC);
        newP = priorP + dP;
        output[i] = newP;
        priorP = newP;
        priorC = getC(newP);
        
    fig = plt.figure(dpi=1000,figsize = [10,8]);
    ax = plt.axes();
    ax.set_xlabel("$Seconds$",fontsize= 14)
    ax.set_ylabel("$Pa$",fontsize= 14)
    #fig.suptitle("Pressure Decline of Fracture with Time",fontsize = 16)
    plt.plot(times[0:sampleNum-1],output[0:sampleNum-1],label = 'Fracture Pressure')
    plt.axhline(resPressure, linestyle='--',color = 'red',label = "Reservoir Pressure")
    plt.legend(prop={'size': 16})
    plt.tight_layout();
    plt.savefig("pressureFalloff.png",dpi=1000);
    plt.show()

#print(deltaP(weekTime,dayTime))

interpolateGraph()