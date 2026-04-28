# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 19:35:26 2023

@author: Spence
"""

import numpy as np;
import random;
from queue import PriorityQueue;
import math;
import matplotlib.pyplot as plt;
import matplotlib.animation as animation;
from matplotlib.colors import ListedColormap
import matplotlib;
from matplotlib.ticker import ScalarFormatter
#random.seed(1);


class ScalarFormatterWithUnit(ScalarFormatter):
    def __init__(self, unit="", *args, **kwargs):
        self.unit = unit
        super().__init__(*args, **kwargs)

    def get_offset(self):
        # Get the normal offset text, e.g. '1e−3' or '×10⁻³'
        offset = super().get_offset()
        if offset:
            return f"{offset} {self.unit}"
        else:
            return self.unit

def __init__():
    print("main!");


def coordAtAngle(length,angle):
    #returning an
    #y, z
    #unadjusted for z coord system, need to translate down.
    return [length*math.cos(angle), length*math.sin(angle)];

pressureVal = 1;
midpointX = 100;
midpointY = 100;
midpointZ = 100;
xLim = 200;
yLim = 200;
zLim = 200;
#pressureGrid = np.zeros(shape = (xLim*yLim));
pressureGrid = np.zeros(shape = (xLim,yLim,zLim));
bondGrid = np.zeros(shape = (xLim,yLim,4)); #bond strength grid. 0 west, 1 east, 2 north, 3 south

bondMaxHorizontal = 100;
bondMaxVertical = 100;

interNodeDist = 1;
faceArea = interNodeDist**2;
surfaceArea = 0;


#midpointX = 50;
#midpointY = 50;
#midpointZ = 50;
#xLim = 100;
#yLim = 100;
#zLim = 100;
bondQueue = 0;
occupiedCells = set();
poppedBonds = set();
createdBonds = set();

injectionX =0;
injectionY =0;
injectionZ =0;
fB =0;
vB =0;

bondMaxX = 1;
bondMaxY = 1;
bondMaxZ = 1;

bondMinX = 0;
bondMinY = 1;
bondMinZ = 0;


def cellIndexer(x,y):
    #y times filled rows + current x val
    if(x<0 or x ==xLim or y<0 or y==yLim):
        return -1;
    return y*xLim + x;

def getCoord(index):
    #x can only range up to xlim
    x = index % xLim; #left over stuff from dividing by index;
    return [int(x),int((index-x)/xLim)];

def getCoord3D(index):
    xy = index % int(xLim * yLim);
    x = xy % xLim;
    return [int(x), int((xy - x)/xLim ) , int((index - xy)/(yLim*xLim))];

def cellIndexer3D(x,y,z):
    if(x<0 or x ==xLim or y<0 or y==yLim or z <0 or z==zLim):
        return -1;
    return z*(xLim*yLim) + y*xLim + x;

def superficialVelocity(frameX,frameY,frameZ):
    #flowRate = .0335; #33.5 L/s .0335 m^3/s
    flowRate = 0.001892705892103
    #flowRate = 0.000015 #15 ml/s (event1)
    
    x = frameX - injectionX;
    y = frameY - injectionY;
    z = frameZ - injectionZ;
    
    #return 0;
    #return flowRate/calculateEllipsoidArea(x, y, z);
    return flowRate/surfaceArea;


def calculatePressureGradient(x,y,z):
    #forchHeimer
    #nonDarcy
    permeabilityRaw = 1.38 * 10**(-20); #https://www.sciencedirect.com/science/article/pii/S0920410519308782#:~:text=According%20to%20Al%20Reda%20et,by%20Al%20Reda%20et%20al.
    #permeability = 1*10**-10
    permeability = 1*10**-11
    nD = 10**8
    #visc = .001006 #a little off for conditions, fix later
    visc =.009;
    #density = 997; #kg/m3 (again fix later)
    density = 1002;
    #density = 0;
    sV = superficialVelocity(x, y, z);
    return - visc/permeability *sV - density/(nD) * sV**2;



def pairingFunction(a,b):
    #going to pair two index values, and return a new number. This will be used to make an easy reference for if a bond exists.
    a0 = a;
    b0 = b;
    if (b < a):
        a0 = b;
        b0 = a;
    
    
    return int(1/2 * (a0 + b0) * (a0 + b0 + 1) + b0);

def invertPairing(z):
    w = math.floor( (math.sqrt(8*z + 1) - 1)/2);
    t = (w**2 + w)/2;
    i2 = z - t; #always the bigger index
    i1 = w - i2;
    
    return [i1,i2];

def checkOccupiedNeighborCount(index):
    #only run after confirming not a redundant bond being popped.
    cellCoord = getCoord3D(index);
    x1 = cellIndexer3D(cellCoord[0]-1, cellCoord[1], cellCoord[2])
    x2 = cellIndexer3D(cellCoord[0]+1, cellCoord[1], cellCoord[2])
    y1 = cellIndexer3D(cellCoord[0], cellCoord[1]+1, cellCoord[2])
    y2 = cellIndexer3D(cellCoord[0], cellCoord[1]-1, cellCoord[2])
    z1 = cellIndexer3D(cellCoord[0], cellCoord[1], cellCoord[2]+1)
    z2 = cellIndexer3D(cellCoord[0], cellCoord[1], cellCoord[2]-1)
    #returns for a given cell how many occupied neighbors it has in each direction
    return [int(x1 in occupiedCells) + int(x2 in occupiedCells), int(y1 in occupiedCells) + int(y2 in occupiedCells), int(z1 in occupiedCells) + int(z2 in occupiedCells)];

def modifySurfaceArea(index):
    global surfaceArea
    xBias = (bondMaxX - bondMinX);
    yBias = (bondMaxY - bondMinY);
    zBias = (bondMaxZ - bondMinZ);
    #print(xBias,yBias,zBias)
    fullConnection = 6; #assuming we're doing 3d, 4 if 2d;
    neighbors = checkOccupiedNeighborCount(index);
    unbiasedSurfaceArea = False; 
    if(unbiasedSurfaceArea ==True):
        occupiedCount = np.sum(neighbors);
        surfaceArea = surfaceArea +(fullConnection - 2*occupiedCount)*faceArea;
    else:
        #doing biases on surface area val based on dimesional access
        #occ Count is count in x y an dz
        surfaceArea = surfaceArea + xBias * (2 - 2*neighbors[0]) *interNodeDist**2 *faceArea;
        surfaceArea = surfaceArea + yBias * (2 - 2*neighbors[1]) *interNodeDist**2 *faceArea;
        surfaceArea = surfaceArea + zBias * (2 - 2*neighbors[2]) *interNodeDist**2 *faceArea;

def getRandomBondStrength(bMax,bMin):
    #old style random.random() * bondMaxVertical
    return (bMax - bMin) *random.random() + bMin;

def percolationLoop(originX,originY,originZ):
    global pressureGrid;
    #random.seed(4);
    firstPass = True;
    loopCondition = True;
    suffix = "_" + str(0) +".npy";
    
    #removed in sample version, uncomment in version with CFF
    #np.save("CFFVals"+suffix, CFF);
    
    
    #relative odds in both directions
    biasFactorY = 0;
    biasFactorX = 0;
    biasFactorZ = 0;
    bondMaxHorizontal = 1 + biasFactorX;
    bondMaxVertical = 1;
    bondMaxZ = 1 + biasFactorZ;
    fB = (bondMaxVertical - biasFactorX)/bondMaxVertical;
    vB = (bondMaxVertical - biasFactorZ)/bondMaxVertical;

    
    percolationOffsetX = originX;
    percolationOffsetY = originY;
    percolationOffsetZ = originZ;
    #initialPressure = 100;
    #initialPressure = 5.43* 10**2
    # = 5430
    #initialPressure = 3450
    #5E-6
    #initialPressure = 3.45*10**6
    initialPressure = 60 * 10**6;
    #initialPressure = 10
    #initialPressure = 1;
    #initialPressure = 10000;
    ##occupiedCells = set();
    #poppedBonds = set();
    #createdBonds = set();
    
    occupiedCells.clear();
    poppedBonds.clear();
    createdBonds.clear();
    bondQueue = PriorityQueue();
    itr = 0;
    while(loopCondition):
        if(firstPass):
            # print("Initial Setup");
            # aftershockLoop();
            # print("CFF");
            # print(CFF);
            firstPass = False;
            
            #-------------------------------------------------------------
            # numFaultNodes = numNodes;
            indexCurrent = cellIndexer3D(midpointX,midpointY,midpointZ);
            modifySurfaceArea(indexCurrent);
            occupiedCells.add(indexCurrent);
            #entering each bond into the priority queue, with the priority being the bond strength. Output is the node with the bond.
            up = cellIndexer3D(midpointX,midpointY-1,midpointZ);
            down = cellIndexer3D(midpointX,midpointY+1,midpointZ);
            right = cellIndexer3D(midpointX+1,midpointY,midpointZ);
            left = cellIndexer3D(midpointX-1,midpointY,midpointZ);
            posZ = cellIndexer3D(midpointX,midpointY,midpointZ+1);
            negZ = cellIndexer3D(midpointX,midpointY,midpointZ-1);
            #print(up,"",down,"",right,"",left);
            if(up==-1 or down ==-1 or right == -1 or left == -1 or posZ ==-1 or negZ ==-1):
                #new bond canditate is at edge of grid;
                #probably an end condition;
                loopCondition = False;
                break;
            #add bonds
            #bondStrength = bondGrid[midPointX,midPointY][]
            # print("IndexCurrent,",indexCurrent);
            # print("u",pairingFunction(indexCurrent, up));
            # print("d",pairingFunction(indexCurrent, down));
            # print("l",pairingFunction(indexCurrent, left));
            # print("r", pairingFunction(indexCurrent, right));
            
            bondQueue.put((getRandomBondStrength(bondMaxY,bondMinY),pairingFunction(indexCurrent, up)));
            createdBonds.add(pairingFunction(indexCurrent, up));
            bondQueue.put((getRandomBondStrength(bondMaxY,bondMinY),pairingFunction(indexCurrent, down)));
            createdBonds.add(pairingFunction(indexCurrent, down));
            bondQueue.put((getRandomBondStrength(bondMaxX,bondMinX),pairingFunction(indexCurrent, left)));
            createdBonds.add(pairingFunction(indexCurrent, left));
            bondQueue.put((getRandomBondStrength(bondMaxX,bondMinX),pairingFunction(indexCurrent, right)));
            createdBonds.add(pairingFunction(indexCurrent, right));
            bondQueue.put((getRandomBondStrength(bondMaxZ,bondMinZ),pairingFunction(indexCurrent, posZ)));
            createdBonds.add(pairingFunction(indexCurrent, posZ));
            bondQueue.put((getRandomBondStrength(bondMaxZ,bondMinZ),pairingFunction(indexCurrent, negZ)));
            createdBonds.add(pairingFunction(indexCurrent, negZ));
            
            #not adding Nodes here
            #pressure = addInflationNode(midpointX + percolationOffsetX, midpointY+percolationOffsetY,midpointZ+percolationOffsetZ,initialPressure,itr,initial=True);
            
            #print(pressure)
            #targetPressure = priorPressure + self.calculatePressureGradient(x,y,z) * self.interNodeDistance;
            pressureGrid[midpointX,midpointY,midpointZ] = initialPressure #+ calculatePressureGradient(midpointX + percolationOffsetX,midpointY + percolationOffsetY,midpointZ + percolationOffsetZ) * interNodeDist;
            #if(pressureGrid[midpointX,midpointY,midpointZ] ==-1): #done early
            #    print("itr",itr)
            #    return [dispOutput,dispX,dispY,dispZ,seisMag];
                #exit();
        else:
            itr = itr + 1;
            #if(itr % 25 == 0 ):
            if(len(occupiedCells) % 25 ==0):
                print("Cell Count: ",len(occupiedCells));
                suffix = "_" + str(len(occupiedCells)) +".npy";
                #print(np.max(CFF))
                #np.save("CFFVals"+suffix, CFF);
                
                #np.save("occupiedCells_"+str(len(occupiedCells))+".npy", np.array(list(occupiedCells)));
                
                # with open ("occupiedCells_"+str(itr)+".txt",'wb') as f:
                #     pickle.dump(occupiedCells,f);
                #print(CFF)
            #get lowest bond
            lowBond = bondQueue.get()[1];
            #print("lowbond",lowBond);
            poppedBonds.add(lowBond);
            #lowbond is a pair index number. Use invert pairings to get the indicies of what the bond is between.
            bondPair = invertPairing(lowBond);
            newSite = '';
            oldSite = '';
            if( bondPair[0] in occupiedCells and bondPair[1] in occupiedCells): #both are occupied already. Popping a bond between occupied cells. That's all.
                #print("Popping redundant bond between ", bondPair[0], " and ",bondPair[1]);
                pass;
            else:
                if(bondPair[0] in occupiedCells):
                    #other one is new space. going 
                    newSite = bondPair[1];
                    oldSite = bondPair[0];
                else:
                    newSite = bondPair[0];
                    oldSite = bondPair[1];
                
                
                if(oldSite not in occupiedCells):
                    raise Exception("error")
                #now we're making a new site and potentially new bonds.
                occupiedCells.add(newSite);
                #need to check directions and see if they're occupied. One always will be, as that was the one it came from. Iterating all though to simply logic though.
                #print(newSite);
                siteCoord = getCoord3D(newSite);
                oldCoord = getCoord3D(oldSite);
                #print(oldCoord)
                #print(siteCoord);
                modifySurfaceArea(newSite);
                up = cellIndexer3D(siteCoord[0],siteCoord[1]-1,siteCoord[2]);
                down = cellIndexer3D(siteCoord[0],siteCoord[1]+1,siteCoord[2]);
                right = cellIndexer3D(siteCoord[0]+1,siteCoord[1],siteCoord[2]);
                left = cellIndexer3D(siteCoord[0]-1,siteCoord[1],siteCoord[2]);
                posZ = cellIndexer3D(siteCoord[0],siteCoord[1],siteCoord[2]+1);
                negZ = cellIndexer3D(siteCoord[0],siteCoord[1],siteCoord[2]-1);
                priorPressure = pressureGrid[oldCoord[0]][oldCoord[1]][oldCoord[2]];
                temp = priorPressure + calculatePressureGradient(siteCoord[0]*interNodeDist+percolationOffsetX,siteCoord[1]*interNodeDist+percolationOffsetY,siteCoord[2]*interNodeDist+percolationOffsetZ) * interNodeDist;
                if(temp <0):
                    loopCondition = False;
                    break
                pressureGrid[siteCoord[0],siteCoord[1],siteCoord[2]] = temp;
                #print(pressure)
                # if(pressure ==-1): #done early
                #     print("Cellcount",len(occupiedCells))
                #     suffix = "_" + str(len(occupiedCells)) +".npy";
                #     #np.save("CFFVals"+suffix, CFF);
                #     #np.save("occupiedCells_"+str(len(occupiedCells))+".npy", np.array(list(occupiedCells)));
                #     return  [dispOutput,dispX,dispY,dispZ,seisMag];
                #pressureGrid[siteCoord[0],siteCoord[1],siteCoord[2]] = pressure;
                
                #print(up,"",down,"",right,"",left);
                if(up==-1 or down ==-1 or right == -1 or left == -1 or posZ == -1 or negZ == -1):
                    #new bond canditate is at edge of grid;
                    #probably an end condition;
                    loopCondition = False;
                    break;
                if(pairingFunction(newSite,up) not in createdBonds ):
                    bondQueue.put((getRandomBondStrength(bondMaxY,bondMinY),pairingFunction(newSite, up)));
                    createdBonds.add(pairingFunction(newSite,up));
                if(pairingFunction(newSite,down) not in createdBonds ):
                    bondQueue.put((getRandomBondStrength(bondMaxY,bondMinY),pairingFunction(newSite, down)));
                    createdBonds.add(pairingFunction(newSite,down));
                if(pairingFunction(newSite,left) not in createdBonds ):
                    bondQueue.put((getRandomBondStrength(bondMaxX,bondMinX),pairingFunction(newSite, left)));
                    createdBonds.add(pairingFunction(newSite,left));
                if(pairingFunction(newSite,right) not in createdBonds ):
                    bondQueue.put((getRandomBondStrength(bondMaxX,bondMinX),pairingFunction(newSite, right)));
                    createdBonds.add(pairingFunction(newSite,right));
                if(pairingFunction(newSite,posZ) not in createdBonds ):
                    bondQueue.put((getRandomBondStrength(bondMaxZ,bondMinZ),pairingFunction(newSite, posZ)));
                    createdBonds.add(pairingFunction(newSite,posZ));
                if(pairingFunction(newSite,negZ) not in createdBonds ):
                    bondQueue.put((getRandomBondStrength(bondMaxZ,bondMinZ),pairingFunction(newSite, negZ)));
                    createdBonds.add(pairingFunction(newSite,negZ));
                
            if(bondQueue.qsize() == 0):
                loopCondition = False; 
                #break;


#outputGrid = np.zeros(shape = (xLim,yLim,zLim));
#pressureVals = pressureGrid[pressureGrid !=0]
#tempSeed = int(random.random()*10000);
#looks good at 9208
tempSeed = 9208
random.seed(tempSeed);
percolationLoop(0,0,0);

pressureNonZero = pressureGrid[pressureGrid !=0];

maxVal = np.max(pressureNonZero);
minVal = np.min(pressureNonZero);
mean = np.mean(pressureNonZero);

#print(np.sum(pressureGrid)/len(occupiedCells))
outputGrid = np.full(shape = (xLim,yLim,zLim),fill_value = 0);

#percolationLoop(0,0,0);

for indexId in occupiedCells:
    coords = getCoord3D(indexId);
    print(coords);
    outputGrid[coords[0]][coords[1]][coords[2]] = pressureGrid[coords[0]][coords[1]][coords[2]];

Output2D = outputGrid[:,midpointY,:];

fig = plt.figure(dpi=1000,figsize = [10,8]);
ax = plt.axes();
ax.set_xlabel("$m$",fontsize= 10)
ax.set_ylabel("$m$",fontsize= 10)
#fig.suptitle("Fluid injection through a vertical fracture slice",fontsize=16);
#norm = matplotlib.colors.Normalize(vmin=outputGrid.min(), vmax=outputGrid.max() )
#cmap = plt.cm.plasma;
#colors = cmap(norm);

basecmap = plt.cm.plasma;
newcmap = basecmap.copy();
newcmap.set_bad(color='black')

masked = np.ma.masked_where(Output2D ==0, Output2D)
pc = plt.pcolormesh(masked,cmap = newcmap);

formatter = ScalarFormatterWithUnit(unit="Pa", useMathText=True);
cbar = fig.colorbar(pc,aspect = 12)

cbar.ax.set_title('$Pa$',fontsize = 14)

plt.tight_layout();


plt.savefig("injectionGraph_seed_"+str(tempSeed)+".png",dpi=1000);
plt.show()
print("Seed: ",tempSeed);
#np.save(file = "outputGridPerc",arr = outputGrid);

# t = np.arange(1,outputnmb,1);


# def animate(i):
#     if(i % 20 == 0 ):
#        print("Iteration: ",i,"/",outputnmb);
#     #ax.clear();
#     cellindex = outputTime.get(i);
#     coords = getCoord(cellindex);
#     outputGrid[coords[0]][coords[1]] = 1;
#     plt.pcolormesh(outputGrid,cmap = 'Purples');
#     return outputGrid
    
# ani = animation.FuncAnimation(fig, func = animate,interval =1, frames = t,  repeat = True);

# ani.save('sample.gif',fps =60);

#fig.show();
#plt.figsize(10,10);
#plt.pcolormesh(outputGrid,cmap = 'Purples');