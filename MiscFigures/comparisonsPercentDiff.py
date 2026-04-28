# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:20:47 2024

@author: spenc
"""

import OkadaSolutions;
import rectangularFault;
import matplotlib.pyplot as plt; 
import matplotlib.patches as patches;
import numpy as np;
import math;
#import okada


depth = -10000;

def pointGen(itr,L):
    point1 = [];
    point2 = [];
    print(itr);
    print(L)
    if(itr ==0):
        return([[L/2],[L/2]])
    for i in range(2**(itr-1)):
        for j in range(2**(itr-1)):
            point1.append(L * (2*i + 1) * 1/(2**itr));
            point2.append(L * (2*j + 1) * 1/(2**itr));
            #points.append(np.array[L * i * 1/(2**itr),L * j * 1/(2**itr)])
            
    return [point1,point2];



pointTypes = ["strike-dip","tensile"];
dipAngles = [math.pi/2,4*math.pi/9,math.pi * 7/18,math.pi/3,math.pi/4]
#90, 80, 70, 60, 45
rakeAngles = [math.pi/2,0] #strike vs dip


dip = math.pi/2;

faultRect = rectangularFault.rectangularFault(-500, 0, depth-500,dip);

depth = -10000;
nodeList = [];

#$\delta = 90 \degree $
rayAngles = [-math.pi/8,-math.pi/4,-3*math.pi/8];
stringRay = ["$\\dfrac{{\pi}}{{8}}$", "$\\dfrac{{\pi}}{{4}}$","$\\dfrac{{3\pi}}{{8}}$"];

nVal = [0,2,3,4,5,6];

numItr = len(nVal);
numC1 = 1;
numC2 = 1;


numRays = len(rayAngles);


numPoints=50
outputAllPoint = np.zeros(shape=(numItr,numRays,numPoints));
outputRect = np.zeros(shape = (numC1,numC2,numPoints));
itrDist = 1000;
xVals = np.linspace(0,numPoints*itrDist,num = numPoints, endpoint = False );

faultSegmentLength = 1000;

lam = 3*10**10
mu = 3.2*10**10
poisson = lam/(2*(lam+mu));
frictCoeff = .5;
x1 = 0;
y1 = 0;


def deltaR(nodeX,nodeY,nodeZ,distance,measureRayAngle):
    #assuming measurement origin at (faultSeg/2, 0, depth+faultseg/2);
    return np.array([distance * math.cos(measureRayAngle) +(faultSegmentLength//2)- nodeX, distance*math.sin(measureRayAngle),(depth+faultSegmentLength//2)-nodeZ -nodeZ]);
    
loadingFromFile = True;


#nVal = [2,3];
if(not loadingFromFile):
    for itr in range(len(nVal)):
        for rayAngleItr in range(len(rayAngles)):
            points = pointGen(nVal[itr],faultSegmentLength);
            area = (1000**2)/(len(points[0]));
            #area = (1000**2);
            # nodeList = [];
            print("Num Points: ",len(points[0]))
            for i  in range(len(points[0])):
                 nodeList.append(OkadaSolutions.OkadaNode(points[0][i], 0, points[1][i] + depth, "strike-dip", area, lam, mu, dip, 0, 0, 0));
            
            
            #outputPoint = np.zeros(numPoints);
            #outputRect = np.zeros(numPoints);
            for i in range(numPoints):
                for node in range(len(nodeList)):
                    temp = deltaR(nodeList[node].mapCoordX,nodeList[node].mapCoordY,nodeList[node].mapCoordZ,itrDist*(i+1),rayAngles[rayAngleItr]);
                    #outputPoint[i] = outputPoint[i] + nodeList[node].evaluateStressTensor(itrDist*(i+1)*math.cos(angle), itrDist*(i+1) * math.sin(angle), depth)[c1][c2];
                    outputAllPoint[itr][rayAngleItr][i] = outputAllPoint[itr][rayAngleItr][i]+nodeList[node].evaluateStressTensor(temp[0], temp[1], temp[2])[x1][y1];
                    #outputPoint[i] = outputPoint[i] + nodeList[node].evaluateStressTensor(temp[0], temp[1], temp[2])[c1][c2];
                
            
            for i in range(len(outputAllPoint[itr][rayAngleItr])):
                #print(outputPoint[i])
                #outputPoint[i] = outputPoint[i]
                if(outputAllPoint[itr][rayAngleItr][i] != 0):
                    #outputAllPoint[itr][c1][c2][i] = math.log10(abs(outputAllPoint[itr][c1][c2][i]));
                    outputAllPoint[itr][rayAngleItr][i] = abs(outputAllPoint[itr][rayAngleItr][i]);
            
            end = numPoints*itrDist;
            
            
else:
    outputAllPoint = np.load("comparisonOutput.npy");
    


#SOURCE DENSITY COMPARISON
f,axarr = plt.subplots(3,figsize=(6,12));

for itr in range(len(nVal)):
    for rayItr in range(len(rayAngles)):
        if(itr ==0):
            axarr[rayItr].plot(xVals,outputAllPoint[itr][rayItr],label =  "1");
        else:
            axarr[rayItr].plot(xVals,outputAllPoint[itr][rayItr],label =  str(4**(nVal[itr]-1)));


for rayItr in range(len(rayAngles)):
    axarr[rayItr].title.set_text("Projection Angle: "+ stringRay[rayItr]);
    axarr[rayItr].set_yscale('log')
    axarr[rayItr].legend(title = "Source Count");
    axarr[rayItr].set_ylabel('Pressure [$Pa$]')
    
axarr[2].set_xlabel('Distance [$m$]')    
plt.tight_layout();
plt.show()


#SINGLE COMPARISON
f,axarr = plt.subplots(1,figsize=(6,4));
axarr.plot(xVals, (outputAllPoint[2][1] - outputAllPoint[1][1])/outputAllPoint[1][1]);
axarr.set_ylabel('Percent Difference')
axarr[2].set_xlabel('Distance [$m$]')
plt.tight_layout();
plt.show()









print("done")