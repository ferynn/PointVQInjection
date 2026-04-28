# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 05:00:31 2025

@author: spenc
"""

import matplotlib.pyplot as plt;
import numpy as np;
import OkadaSolutions;
import numpy as np;
import math;
import random;
from queue import PriorityQueue;
import sympy;
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import brownian;
import os;
import pickle;
import pandas as pd;
from matplotlib.ticker import MaxNLocator
import geometryGen as gG;

import nodeManager;

class Simulation:

    def __init__(self, eventNumber):
        self.eventNumber =eventNumber; #how many times to run the simulation

        # self.surfaceArea = 0;
        self.flow = .1;
        #self.numBursts = numBursts
        self.numBursts =1;
        self.xLim = 0
        self.yLim = 0
        self.zLim = 0
        self.midpointX = self.xLim//2
        self.midpointY = self.yLim//2
        self.midpointZ = self.zLim//2
        self.bondQueue = 0;
        self.occupiedCells = [set() for _ in range(self.numBursts)];
        self.poppedBonds = [set() for _ in range(self.numBursts)];
        self.createdBonds = [set() for _ in range(self.numBursts)];
        self.pressureGrid = 0; 
        self.percolationOffsetX = np.zeros(self.numBursts);
        self.percolationOffsetY = np.zeros(self.numBursts);
        self.percolationOffsetZ = np.zeros(self.numBursts);
        self.interNodeDistance = np.zeros(self.numBursts);
        self.faceArea = np.zeros(self.numBursts);
        
        self.injectionX =np.zeros(self.numBursts);
        self.injectionY =np.zeros(self.numBursts);
        self.injectionZ =np.zeros(self.numBursts);
        
        #self.magOutput = np.zeros(eventNumber);
    
        #xi0 = dc/v0
        #random.seed(0);

        self.finalPercolationBenchmark = False;
        self.nodePastTarget = 0;
        self.nodesPastMult = 2;
        self.currentNodesPast = 0;
        
        self.nodeTarget =50;
        
        self.bondMaxX = 1;
        self.bondMaxY = 1;
        self.bondMaxZ = .8;

        self.bondMinX = 0;
        self.bondMinY = .37;
        self.bondMinZ = 0;
        
        self.porosity = .3;
        
        
        self.surfaceArea = np.zeros(self.numBursts);
        self.initialPressure = np.zeros(self.numBursts);
        self.percolationPreCheck = False;
        
        self.hitEdge = False;
        
        self.NodeManager = nodeManager.NodeManager(eventNumber);


    def coordAtAngle(self, length,angle):
        #returning an
        #y, z
        #unadjusted for z coord system, need to translate down.
        return [length*math.cos(angle), length*math.sin(angle)];
    
    def cellIndexer(self, x,y):
        #y times filled rows + current x val
        if(x<0 or x ==self.xLim or y<0 or y==self.yLim):
            return -1;
        return y*self.xLim + x;
    
    def cellIndexer3D(self, x,y,z):
        if(x<0 or x ==self.xLim or y<0 or y==self.yLim or z <0 or z==self.zLim):
            return -1; 
        return z*(self.xLim*self.yLim) + y*self.xLim + x;
    
    def getCoord(self,index):
        #x can only range up to xlim
        x = index % self.xLim; #left over stuff from dividing by index;
        return [int(x),int((index-x)/self.xLim)];
    
    
    def getCoord3D(self,index):
        xy = index % int(self.xLim * self.yLim);
        x = xy % self.xLim;
        return [int(x), int((xy - x)/self.xLim ) , int((index - xy)/(self.yLim*self.xLim))];

    def pairingFunction(self,a,b):
        #going to pair two index values, and return a new number. This will be used to make an easy reference for if a bond exists.
        a0 = a;
        b0 = b;
        if (b < a):
            a0 = b;
            b0 = a;
    
        
        return int(1/2 * (a0 + b0) * (a0 + b0 + 1) + b0);


    def invertPairing(self,z):
        w = math.floor( (math.sqrt(8*z + 1) - 1)/2);
        t = (w**2 + w)/2;
        i2 = z - t; #always the bigger index
        i1 = w - i2;
        
        return [i1,i2];

    #def setInflationPressure(self, Node):
    
        
    def calculateEllipsoidArea(self,frameX,frameY,frameZ):
        y = frameX;
        x = frameY; #swapping to the right vals for the equation. The X axis in our normal setup is the "y" variable in the ellipsoid calcs
        z = frameZ;
        #faultBias is multiplier of fault direction bond strength vs orthogonal bond strength (kept as one)
        #verticalBias is muliplier of vertical bond strnge vs orthogonal bond strength
        a = math.sqrt(self.fB**2 * (self.vB**2 * x**2 + z**2) + self.vB**2 * y**2)/(self.vB * self.fB);
        
        #using ellipsoid approximation
        p=1.6075;
        return  4*math.pi * (((a**p * (self.fB*a)**p) + (a**p *(self.vB*a)**p) + ((self.fB*a)**p * (self.vB*a)**p))/3)**(1/p);
        
        #return (a,res);
    
    def superficialVelocity(self,frameX,frameY,frameZ,burstId):
        return self.flow/self.surfaceArea[burstId]
    
    def checkOccupiedNeighborCount(self,index,burstId):
        #only run after confirming not a redundant bond being popped.
        cellCoord = self.getCoord3D(index);
        x1 = self.cellIndexer3D(cellCoord[0]-1, cellCoord[1], cellCoord[2])
        x2 = self.cellIndexer3D(cellCoord[0]+1, cellCoord[1], cellCoord[2])
        y1 = self.cellIndexer3D(cellCoord[0], cellCoord[1]+1, cellCoord[2])
        y2 = self.cellIndexer3D(cellCoord[0], cellCoord[1]-1, cellCoord[2])
        z1 = self.cellIndexer3D(cellCoord[0], cellCoord[1], cellCoord[2]+1)
        z2 = self.cellIndexer3D(cellCoord[0], cellCoord[1], cellCoord[2]-1)
        #returns for a given cell how many occupied neighbors it has in each direction
        return [int(x1 in self.occupiedCells[burstId]) + int(x2 in self.occupiedCells[burstId]), int(y1 in self.occupiedCells[burstId]) + int(y2 in self.occupiedCells[burstId]), int(z1 in self.occupiedCells[burstId]) + int(z2 in self.occupiedCells[burstId])];

    def modifySurfaceArea(self,burstId,index):
        xBias = (self.bondMaxX - self.bondMinX);
        yBias = (self.bondMaxY - self.bondMinY);
        zBias = (self.bondMaxZ - self.bondMinZ);
        #print(xBias,yBias,zBias)
        fullConnection = 6; #assuming we're doing 3d, 4 if 2d;
        neighbors = self.checkOccupiedNeighborCount(index,burstId);
        unbiasedSurfaceArea = False; 
        if(unbiasedSurfaceArea ==True):
            occupiedCount = np.sum(neighbors);
            self.surfaceArea[burstId] = self.surfaceArea[burstId] +(fullConnection - 2*occupiedCount)*self.faceArea[burstId];
        else:
            #doing biases on surface area val based on dimesional access
            #occ Count is count in x y an dz
            self.surfaceArea[burstId] = self.surfaceArea[burstId] + xBias * (2 - 2*neighbors[0]) *self.faceArea[burstId];
            self.surfaceArea[burstId] = self.surfaceArea[burstId] + yBias * (2 - 2*neighbors[1]) *self.faceArea[burstId];
            self.surfaceArea[burstId] = self.surfaceArea[burstId] + zBias * (2 - 2*neighbors[2]) *self.faceArea[burstId];

    def getRandomBondStrength(self,bMax,bMin):
        #old style random.random() * bondMaxVertical
        return (bMax - bMin) *random.random() + bMin;

    def calculatePressureGradient(self,x,y,z,burstId):
        #forchHeimer
        #nonDarcy
        permeabilityRaw = 1.38 * 10**(-20); #https://www.sciencedirect.com/science/article/pii/S0920410519308782#:~:text=According%20to%20Al%20Reda%20et,by%20Al%20Reda%20et%20al.
        #permeability = 1*10**-10
        permeability = 1*10**-7
        nD = 10**8
        visc = .001006 #a little off for conditions, fix later
        density = 997; #kg/m3 (again fix later)
        #density = 0;
        sV = self.superficialVelocity(x, y, z,burstId);
        return - visc/permeability *sV - density/(nD) * sV**2;
    
    
    def getInflationMag(self, node,dist,multi):
        temp1 = np.diagonal(multi * node.getStressTensorAt(0,0,node.mapCoordZ + dist));
        temp2 = np.diagonal(multi * node.getStressTensorAt(0,0,node.mapCoordZ - dist));
        #print("rawmag")
        #print(((np.linalg.norm(temp1)+np.linalg.norm(temp2))/2))
        return ((np.linalg.norm(temp1)+np.linalg.norm(temp2))/2);
    
    def setStressToPressure(self, node,pressure,tolerance,checkAtDist,knownMultiplier = -1):
        multi=1;
        #print(checkAtDist)
        if(knownMultiplier != -1):
            multi = knownMultiplier;
            currentPressure = self.getInflationMag(node,checkAtDist, multi);
        else:
            currentPressure = self.getInflationMag(node, checkAtDist,multi);
        while(not math.isclose(currentPressure, pressure,rel_tol = tolerance)):
            multi = multi * (pressure/currentPressure);
            currentPressure = self.getInflationMag(node, checkAtDist,multi);
        #print((currentPressure,multi))
        return (currentPressure,multi)
            
    
    def percolationCFFCheck(self,itr):
        done = False;
        #print(self.numCFFNodes)
        #going a little past the final amount for one event, to amp it
        for i in range(self.NodeManager.numCFFNodes):
            if(self.CFF[i] >0):
                if(self.finalPercolationBenchmark == False):
                    self.finalPercolationBenchmark = True;
                    #self.nodesPastTarget = int(self.nodesPastMult * itr);
                    sigmaLogNormal =1.25;
                    logVal = np.random.lognormal(math.log(self.nodesPastMult),sigmaLogNormal,1);
                    if(logVal < 1):
                        logVal =1;
                    self.nodesPastTarget = int(logVal*itr);
                    if(self.nodesPastTarget <= itr):
                        self.nodesPastTarget = itr+1;
                        #safety check
                    
                    #self.nodesPastTarge = int(self.nodesPastMult * itr)
                    print("Current Itr ",itr, "Projected Final Itr ",self.nodesPastTarget);
        if(self.finalPercolationBenchmark==True):
            if(self.nodesPastTarget <= itr):
                done = True;
        return done;
        
    def addInflationNode(self,x,y,z,priorPressure,nodeItr,burstId,initial=False):
        #need to generate node, modify greens arrays
        #zVal = self.percolationOff;
        #self.addNode(OkadaSolutions.Node(x,y,zVal,"inflation",1,1,300.2*10**10, 300*10**10, 0,0,0));
        
        #porosity = .21; #porosity of opalinus clay according to that one study.
        effectiveVol = (self.interNodeDistance[burstId])**3 *self.porosity; #each cell is a little cube of volume, filled according to the porosity fraction.
        targetPressure =0;
        if(initial):
            targetPressure = priorPressure; #first one is just set to a fixed val.
        else:
            targetPressure = priorPressure + self.calculatePressureGradient(x,y,z,burstId) * self.interNodeDistance; #quick, dirty, not quite correct. Fix angle.
            if(targetPressure <=0):
                targetPressure = 0;
        if(nodeItr % 50 == 0):
            #print(nodeItr," Pressure: ",targetPressure);
            print("Node: ",nodeItr);
            #self.update_CFF();
            #print("max ",np.max(self.CFF))
        testFactor = 5;
        #print(inflationMagnitude)
        inflationMagnitude = effectiveVol * targetPressure*testFactor; #gives correct N*m units for M0; 
        #print(inflationMagnitude)
        #print(inflationMagnitude)
        lambd = 3.722*10**9;
        mu = 1.2*10**9;
        tempNode = OkadaSolutions.OkadaNode(x,y,z,"inflation",inflationMagnitude,lambd, mu, 0,0,0,burstId);
        
        
        tempStressNormal= np.zeros(self.NodeManager.numNodes);
        tempStressShear= np.zeros(self.NodeManager.numNodes);
        for i in range(self.NodeManager.numFaultNodes):#iterating over the non-inflation nodes.
            #going to 
            stress_temp = tempNode.getStress(self.NodeManager.nodeList[i]);
            if(stress_temp[0]==0 and stress_temp[1]==0):
                raise Exception("zero stress on node error");
            tempStressNormal[i] = stress_temp[0];
            tempStressShear[i] = stress_temp[1];
        
        # for i in range(self.numFaultNodes):
        #     self.normalGreens[self.numFaultNodes+nodeItr][i] = tempStressNormal[i];
        #     self.shearGreens[self.numFaultNodes+nodeItr][i] = tempStressShear[i];
            
        for i in range(self.NodeManager.numFaultNodes):
            self.NodeManager.normalGreens[i][self.NodeManager.numFaultNodes+nodeItr] = tempStressNormal[i];
            self.NodeManager.shearGreens[i][self.NodeManager.numFaultNodes+nodeItr] = tempStressShear[i];
            
            
        #slipCalc = [1,1];
        slipMulti =1;

        self.NodeManager.slipMagShear[self.NodeManager.numCFFNodes + nodeItr] = slipMulti;
        #self.slipMagShear= np.append(self.slipMagShear, 1);
        
        #print("Added Node at index ",self.numNodes);
        self.NodeManager.addNode(tempNode);

        return targetPressure;
        
        
    def preInitPercLoop(self,interNodeDist,numBursts):
        self.percolationPreCheck = True;
        d = interNodeDist;
        self.interNodeDistance.fill(d);
        self.faceArea.fill(d**2);
        self.initialPressure[0] = 69 * 10**6;
        self.injectionX[0] = 300;
        self.injectionY[0] = 100;
        self.injectionZ[0] = 500;
        self.percolationOffsetX[0] = 0;
        self.percolationOffsetY[0] = 0;
        self.percolationOffsetZ[0] = 0;
        self.xLim = 1000;
        self.yLim = 200;
        self.zLim = 1000;
        self.pressureGrid = np.zeros(shape = (self.numBursts,self.xLim,self.yLim,self.zLim));
        
    def remove_from_priority_queue(self,pq,targetBond): #removing non-lowest bond strength bond from priority queue
        temp = [];
        while not (pq.empty()):
            item = pq.get() #get lowest priority bond
            if item[1] != targetBond:
                temp.append(item)
        for item in temp:
            pq.put(item)
        return pq;
        
    def calcNumCellsFromTime(self,time, flow, porosity,burstId):
        #time in seconds, flow in m^3/sec, porosity
        cellFluidVol = porosity * self.interNodeDistance[burstId]**3;
        totalVol = flow*time;
        return totalVol/cellFluidVol;
        
    def percolationLoop(self,originX,originY,originZ,burstId,seed):
        if(self.percolationPreCheck == False):
            raise Exception("Percolation not initalized.");
        #random.seed(4);
        random.seed(seed);
        firstPass = True;
        loopCondition = True;
        suffix = "_" + str(0) +".npy";
        np.save("CFFVals"+suffix, self.CFF);
        
        #injectTime = 10800; #3 hour injection
        injectTime = 5*3600
        furthestBond = 0;
        furthestX =0;
        self.bondQueue = PriorityQueue();
        itr = 0;
        cellCountTarget= int(self.calcNumCellsFromTime(injectTime, self.flow, self.porosity, burstId))
        print("cellCountTarget:",cellCountTarget)
        cellCount = 0;
        frackCycleCount = 10;
        fileOutputGrid = np.full(shape = (frackCycleCount,cellCountTarget*frackCycleCount),fill_value=-1);
        for fracPropagationItr in range(frackCycleCount):
            print("propItr", fracPropagationItr)
            cellCount =0;
            loopCondition = True;
            firstPass = True;
            #print(loopCondition)
            while(loopCondition):
                if(firstPass and fracPropagationItr==0): #true first iteration
                    # print("Initial Setup");
                    # self.aftershockLoop();
                    # print("CFF");
                    # print(self.CFF);
                    firstPass = False;
                    
                    #-------------------------------------------------------------
                    # self.numFaultNodes = self.numNodes;
                    indexCurrent = self.cellIndexer3D(self.injectionX[burstId],self.injectionY[burstId],self.injectionZ[burstId]);
                    self.modifySurfaceArea(burstId,indexCurrent);
                    self.occupiedCells[burstId].add(indexCurrent);
                    #entering each bond into the priority queue, with the priority being the bond strength. Output is the node with the bond.
                    up = self.cellIndexer3D(self.injectionX[burstId],self.injectionY[burstId]-1,self.injectionZ[burstId]);
                    down = self.cellIndexer3D(self.injectionX[burstId],self.injectionY[burstId]+1,self.injectionZ[burstId]);
                    right = self.cellIndexer3D(self.injectionX[burstId]+1,self.injectionY[burstId],self.injectionZ[burstId]);
                    left = self.cellIndexer3D(self.injectionX[burstId]-1,self.injectionY[burstId],self.injectionZ[burstId]);
                    posZ = self.cellIndexer3D(self.injectionX[burstId],self.injectionY[burstId],self.injectionZ[burstId]+1);
                    negZ = self.cellIndexer3D(self.injectionX[burstId],self.injectionY[burstId],self.injectionZ[burstId]-1);
                    #print(up,"",down,"",right,"",left);
                    if(up==-1 or down ==-1 or right == -1 or left == -1 or posZ ==-1 or negZ ==-1):
                        #new bond canditate is at edge of grid;
                        #probably an end condition;
                        loopCondition = False;
                        stopAll = True;
                        break;
                    #add bonds
                    
                    cellCount = cellCount +1;
                    
                    furthestBond = self.pairingFunction(indexCurrent, right);
                    furthestX = self.injectionX[burstId]; #track first X coord of this burst, as we propgate it horizontally
                    #print("furthestX: ",furthestX);
                    #print("CellCoord ", self.getCoord3D(right));
                    
                    self.bondQueue.put((self.getRandomBondStrength(self.bondMaxY,self.bondMinY),self.pairingFunction(indexCurrent, up)));
                    self.createdBonds[burstId].add(self.pairingFunction(indexCurrent, up));
                    self.bondQueue.put((self.getRandomBondStrength(self.bondMaxY,self.bondMinY),self.pairingFunction(indexCurrent, down)));
                    self.createdBonds[burstId].add(self.pairingFunction(indexCurrent, down));
                    self.bondQueue.put((self.getRandomBondStrength(self.bondMaxX,self.bondMinX),self.pairingFunction(indexCurrent, left)));
                    self.createdBonds[burstId].add(self.pairingFunction(indexCurrent, left));
                    self.bondQueue.put((self.getRandomBondStrength(self.bondMaxX,self.bondMinX),self.pairingFunction(indexCurrent, right)));
                    self.createdBonds[burstId].add(self.pairingFunction(indexCurrent, right));
                    self.bondQueue.put((self.getRandomBondStrength(self.bondMaxZ,self.bondMinZ),self.pairingFunction(indexCurrent, posZ)));
                    self.createdBonds[burstId].add(self.pairingFunction(indexCurrent, posZ));
                    self.bondQueue.put((self.getRandomBondStrength(self.bondMaxZ,self.bondMinZ),self.pairingFunction(indexCurrent, negZ)));
                    self.createdBonds[burstId].add(self.pairingFunction(indexCurrent, negZ));
                    
                    #pressure = self.addInflationNode(self.injectionX[burstId] + self.percolationOffsetX[burstId], self.injectionY[burstId]+self.percolationOffsetY[burstId],self.injectionZ[burstId]+self.percolationOffsetZ[burstId],self.initialPressure[burstId],itr,initial=True);
                    pressure = self.initialPressure[burstId];
                    
                    #print(pressure)
                    #print(burstId);
                    #print(self.injectionX[burstId]," ",self.injectionY[burstId]," ",self.injectionZ[burstId])
                    #print(self.pressureGrid.shape)
                    self.pressureGrid[burstId][int(self.injectionX[burstId])][int(self.injectionY[burstId])][int(self.injectionZ[burstId])] = pressure;
                    #print(pressure)
                    if(pressure <0 or cellCount == cellCountTarget): #done early
                        # print("itr",itr)
                        print("test")
                        # return [self.dispOutput,self.dispX,self.dispY,self.dispZ,self.seisMag];
                        loopCondition = False;
                        break;
                        #exit();
                elif(firstPass and fracPropagationItr!=0): #instead of starting at the inital injection site, start again by expanding into the right most bond
                    # print("Initial Setup");
                    # self.aftershockLoop();
                    # print("CFF");
                    firstPass = False;
                    self.bondQueue = self.remove_from_priority_queue(self.bondQueue,furthestBond); #popping a bond out of order.
                    self.poppedBonds[burstId].add(furthestBond);
                    bondPair = self.invertPairing(furthestBond);
                    newSite = '';
                    oldSite = '';
                    if(bondPair[0] in self.occupiedCells[burstId]):
                        #other one is new space. going 
                        newSite = bondPair[1];
                        oldSite = bondPair[0];
                    else:
                        newSite = bondPair[0];
                        oldSite = bondPair[1];
                        
                    self.occupiedCells[burstId].add(newSite);
                    
                    siteCoord = self.getCoord3D(newSite);
                    oldCoord = self.getCoord3D(oldSite);
                    
                    self.modifySurfaceArea(burstId,newSite);
                        
                    up = self.cellIndexer3D(siteCoord[0],siteCoord[1]-1,siteCoord[2]);
                    down = self.cellIndexer3D(siteCoord[0],siteCoord[1]+1,siteCoord[2]);
                    right = self.cellIndexer3D(siteCoord[0]+1,siteCoord[1],siteCoord[2]);
                    left = self.cellIndexer3D(siteCoord[0]-1,siteCoord[1],siteCoord[2]);
                    posZ = self.cellIndexer3D(siteCoord[0],siteCoord[1],siteCoord[2]+1);
                    negZ = self.cellIndexer3D(siteCoord[0],siteCoord[1],siteCoord[2]-1);
                    priorPressure = self.pressureGrid[burstId][oldCoord[0]][oldCoord[1]][oldCoord[2]];
                    
                    if(siteCoord[0]>furthestX): #storing the furthest occupied cell in the x direction;
                        furthestX = siteCoord[0];
                        furthestBond = self.pairingFunction(newSite, right);
                    
                    pressure = priorPressure + self.calculatePressureGradient(siteCoord[0]*self.interNodeDistance[burstId]+self.percolationOffsetX[burstId],siteCoord[1]*self.interNodeDistance[burstId]+self.percolationOffsetY[burstId],siteCoord[2]*self.interNodeDistance[burstId]+self.percolationOffsetZ[burstId],burstId) * self.interNodeDistance[burstId];
                    if(pressure ==0): #done early
                        #print("Cellcount",len(self.occupiedCells[burstId]))
                        #suffix = "_" + str(len(self.occupiedCells[burstId])) +".npy";
                        #np.save("CFFVals"+suffix, self.CFF);
                        #np.save("occupiedCells_"+str(len(self.occupiedCells))+".npy", np.array(list(self.occupiedCells)));
                        #return  [self.dispOutput,self.dispX,self.dispY,self.dispZ,self.seisMag];
                        #stopAll = True;
                        #loopCondition =False;
                        return;
                    self.pressureGrid[burstId][siteCoord[0]][siteCoord[1]][siteCoord[2]] = pressure;
                    #print(up,"",down,"",right,"",left);
                    if(up==-1 or down ==-1 or right == -1 or left == -1 or posZ == -1 or negZ == -1):
                        #new bond canditate is at edge of grid;
                        #probably an end condition;
                        loopCondition = False;
                        print("cellCount:",cellCount);
                        #raise Exception("Error hit edge");
                        self.hitEdge = True;
                        return;
                    cellCount = cellCount+1;
                    if(cellCountTarget == cellCount):
                        loopCondition = False;
                        print("furthestX: ",furthestX);
                        print("CellCoord ", self.getCoord3D(self.invertPairing(furthestBond)[1]));
                        break
                    if(self.pairingFunction(newSite,up) not in self.createdBonds[burstId] ):
                        self.bondQueue.put((self.getRandomBondStrength(self.bondMaxY,self.bondMinY),self.pairingFunction(newSite, up)));
                        self.createdBonds[burstId].add(self.pairingFunction(newSite,up));
                    if(self.pairingFunction(newSite,down) not in self.createdBonds[burstId] ):
                        self.bondQueue.put((self.getRandomBondStrength(self.bondMaxY,self.bondMinY),self.pairingFunction(newSite, down)));
                        self.createdBonds[burstId].add(self.pairingFunction(newSite,down));
                    if(self.pairingFunction(newSite,left) not in self.createdBonds[burstId] ):
                        self.bondQueue.put((self.getRandomBondStrength(self.bondMaxX,self.bondMinX),self.pairingFunction(newSite, left)));
                        self.createdBonds[burstId].add(self.pairingFunction(newSite,left));
                    if(self.pairingFunction(newSite,right) not in self.createdBonds[burstId] ):
                        self.bondQueue.put((self.getRandomBondStrength(self.bondMaxX,self.bondMinX),self.pairingFunction(newSite, right)));
                        self.createdBonds[burstId].add(self.pairingFunction(newSite,right));
                    if(self.pairingFunction(newSite,posZ) not in self.createdBonds[burstId] ):
                        self.bondQueue.put((self.getRandomBondStrength(self.bondMaxZ,self.bondMinZ),self.pairingFunction(newSite, posZ)));
                        self.createdBonds[burstId].add(self.pairingFunction(newSite,posZ));
                    if(self.pairingFunction(newSite,negZ) not in self.createdBonds[burstId] ):
                        self.bondQueue.put((self.getRandomBondStrength(self.bondMaxZ,self.bondMinZ),self.pairingFunction(newSite, negZ)));
                        self.createdBonds[burstId].add(self.pairingFunction(newSite,negZ));
                    
                    
                else:
                    itr = itr + 1;

                    #get lowest bond
                    #print("itr",fracPropagationItr);
                    lowBond = self.bondQueue.get()[1];
                    #print("lowbond",lowBond);
                    self.poppedBonds[burstId].add(lowBond);
                    #lowbond is a pair index number. Use invert pairings to get the indicies of what the bond is between.
                    bondPair = self.invertPairing(lowBond);
                    newSite = '';
                    oldSite = '';
                    if( bondPair[0] in self.occupiedCells[burstId] and bondPair[1] in self.occupiedCells[burstId]): #both are occupied already. Popping a bond between occupied cells. That's all.
                        #print("Popping redundant bond between ", bondPair[0], " and ",bondPair[1]);
                        pass;
                    else:
                        if(bondPair[0] in self.occupiedCells[burstId]):
                            #other one is new space. going 
                            newSite = bondPair[1];
                            oldSite = bondPair[0];
                        else:
                            newSite = bondPair[0];
                            oldSite = bondPair[1];

                        self.occupiedCells[burstId].add(newSite);
                        #need to check directions and see if they're occupied. One always will be, as that was the one it came from. Iterating all though to simply logic though.
                        #print(newSite);
                        siteCoord = self.getCoord3D(newSite);
                        oldCoord = self.getCoord3D(oldSite);
                        #print(oldCoord)
                        #print(siteCoord);
                        self.modifySurfaceArea(burstId,newSite);
                        
                            
                        up = self.cellIndexer3D(siteCoord[0],siteCoord[1]-1,siteCoord[2]);
                        down = self.cellIndexer3D(siteCoord[0],siteCoord[1]+1,siteCoord[2]);
                        right = self.cellIndexer3D(siteCoord[0]+1,siteCoord[1],siteCoord[2]);
                        left = self.cellIndexer3D(siteCoord[0]-1,siteCoord[1],siteCoord[2]);
                        posZ = self.cellIndexer3D(siteCoord[0],siteCoord[1],siteCoord[2]+1);
                        negZ = self.cellIndexer3D(siteCoord[0],siteCoord[1],siteCoord[2]-1);
                        priorPressure = self.pressureGrid[burstId][oldCoord[0]][oldCoord[1]][oldCoord[2]];
                        if(siteCoord[0]>furthestX): #storing the furthest occupied cell in the x direction;
                            furthestX = siteCoord[0];
                            furthestBond = self.pairingFunction(newSite, right);
                        
                        #print("PP",priorPressure)
                        pressure = priorPressure + self.calculatePressureGradient(siteCoord[0]*self.interNodeDistance[burstId]+self.percolationOffsetX[burstId],siteCoord[1]*self.interNodeDistance[burstId]+self.percolationOffsetY[burstId],siteCoord[2]*self.interNodeDistance[burstId]+self.percolationOffsetZ[burstId],burstId) * self.interNodeDistance[burstId];
                        #print(pressure)
                        if(pressure ==0): #done early
                            #print("Cellcount",len(self.occupiedCells[burstId]))
                            #suffix = "_" + str(len(self.occupiedCells[burstId])) +".npy";
                            #np.save("CFFVals"+suffix, self.CFF);
                            #np.save("occupiedCells_"+str(len(self.occupiedCells))+".npy", np.array(list(self.occupiedCells)));
                            #return  [self.dispOutput,self.dispX,self.dispY,self.dispZ,self.seisMag];
                            loopCondition = False;
                            return;
                        self.pressureGrid[burstId][siteCoord[0]][siteCoord[1]][siteCoord[2]] = pressure;
                        #print(up,"",down,"",right,"",left);
                        if(up==-1 or down ==-1 or right == -1 or left == -1 or posZ == -1 or negZ == -1):
                            #new bond canditate is at edge of grid;
                            #probably an end condition;
                            print("cellCount:",cellCount);
                            print("hitEdgeDiscarding",cellCount);
                            #raise Exception("Error hit edge");
                            loopCondition = False;
                            self.hitEdge = True;
                            return;
                        cellCount = cellCount+1;
                        if(cellCountTarget == cellCount):
                            loopCondition = False;
                            print("furthestX: ",furthestX);
                            print("CellCoord ", self.getCoord3D(self.invertPairing(furthestBond)[1]));
                            break;
                        if(self.pairingFunction(newSite,up) not in self.createdBonds[burstId] ):
                            self.bondQueue.put((self.getRandomBondStrength(self.bondMaxY,self.bondMinY),self.pairingFunction(newSite, up)));
                            self.createdBonds[burstId].add(self.pairingFunction(newSite,up));
                        if(self.pairingFunction(newSite,down) not in self.createdBonds[burstId] ):
                            self.bondQueue.put((self.getRandomBondStrength(self.bondMaxY,self.bondMinY),self.pairingFunction(newSite, down)));
                            self.createdBonds[burstId].add(self.pairingFunction(newSite,down));
                        if(self.pairingFunction(newSite,left) not in self.createdBonds[burstId] ):
                            self.bondQueue.put((self.getRandomBondStrength(self.bondMaxX,self.bondMinX),self.pairingFunction(newSite, left)));
                            self.createdBonds[burstId].add(self.pairingFunction(newSite,left));
                        if(self.pairingFunction(newSite,right) not in self.createdBonds[burstId] ):
                            self.bondQueue.put((self.getRandomBondStrength(self.bondMaxX,self.bondMinX),self.pairingFunction(newSite, right)));
                            self.createdBonds[burstId].add(self.pairingFunction(newSite,right));
                        if(self.pairingFunction(newSite,posZ) not in self.createdBonds[burstId] ):
                            self.bondQueue.put((self.getRandomBondStrength(self.bondMaxZ,self.bondMinZ),self.pairingFunction(newSite, posZ)));
                            self.createdBonds[burstId].add(self.pairingFunction(newSite,posZ));
                        if(self.pairingFunction(newSite,negZ) not in self.createdBonds[burstId] ):
                            self.bondQueue.put((self.getRandomBondStrength(self.bondMaxZ,self.bondMinZ),self.pairingFunction(newSite, negZ)));
                            self.createdBonds[burstId].add(self.pairingFunction(newSite,negZ));
                        
                    if(self.bondQueue.qsize() == 0):
                        raise Exception("This shouldn't happen")
                        loopCondition = False; 
            print("Copying")
            saveItr = 0;
            for cell in self.occupiedCells[burstId]:
                fileOutputGrid[fracPropagationItr][saveItr] = cell;
                saveItr = saveItr + 1;
        np.save("C:\\Users\\spenc\\Documents\\OkadaPoint\\burstData\\"+str(seed)+"_"+str(burstId)+".npy",fileOutputGrid);








#d = 3;


#BIGGGGG
interNodeDist = 5;


xLim = 1000;
yLim = 200;
zLim = 1000;
#self.pressureGrid = np.zeros(shape = (self.numBursts,self.xLim,self.yLim,self.zLim));


def getCoord3D(index):
    xy = index % int(xLim * yLim);
    x = xy % xLim;
    return [int(x), int((xy - x)/xLim ) , int((index - xy)/(yLim*xLim))];


def burstStuff():
    seedVal = 1063;
    cells = np.load("C:\\Users\\spenc\\Documents\\OkadaPoint\\burstData\\" + str(seedVal)+"_0.npy");
    
    testGrid = np.zeros(shape = (xLim,yLim,zLim));
    
    numItrations = cells.shape[0];
    maxCells = cells.shape[1]
    
    maxX = -1;
    minX = 1000000;
    
    maxZ = -1;
    minZ = 1000000;
    
    maxY = -1;
    minY = 1000000;
    
    currentIteration = 4;
    
    for i in range(maxCells):
        if(cells[currentIteration][i] == -1):
            print("NumCells "+ str(i+1));
            break;
        coord = getCoord3D(cells[currentIteration][i]);
        if(coord[0] < minX):
            minX = coord[0];
        if(coord[0] > maxX):
            maxX = coord[0];
            
        if(coord[1] < minY):
            minY = coord[1];
        if(coord[1] > maxY):
            maxY = coord[1];
            
        if(coord[2] < minZ):
            minZ = coord[2];
        if(coord[2] > maxZ):
            maxZ = coord[2];
        #testGrid[coord[0]][coord[1]][coord[2]] = 1;
    
    testGrid = np.zeros(shape = (maxX -minX +1,maxY -minY +1,maxZ -minZ +1));
    
    for i in range(maxCells):
        if(cells[currentIteration][i] == -1):
            print("NumCells "+ str(i+1));
            break;
            
        coord = getCoord3D(cells[currentIteration][i]);
        testGrid[coord[0]-minX][coord[1] - minY][coord[2]-minZ] = 1;
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(testGrid)
    plt.show()

def getVoxelCoord(rawGrid,injectionDispX, injectionDispY,injectionDispZ,interNodeDist):
    tempX = 0
    tempY = 1;
    tempZ= 2;
    coordGridX = np.zeros(shape = (rawGrid.shape[tempX] +1,rawGrid.shape[tempY] +1,rawGrid.shape[tempZ] +1));
    coordGridY = np.zeros(shape = (rawGrid.shape[tempX] +1,rawGrid.shape[tempY] +1,rawGrid.shape[tempZ] +1));
    coordGridZ = np.zeros(shape = (rawGrid.shape[tempX] +1,rawGrid.shape[tempY] +1,rawGrid.shape[tempZ] +1));
    for x in range(len(coordGridX)):
        for y in range(len(coordGridX[0])):
            for z in range(len(coordGridX[0][0])):
                coordGridX[x][y][z] = injectionDispX + x * interNodeDist
                coordGridY[x][y][z] = injectionDispY + y * interNodeDist
                coordGridZ[x][y][z] = injectionDispZ + z * interNodeDist

                
    return [coordGridX,coordGridY,coordGridZ];
    

seed = 35
random.seed(seed);
np.random.seed(seed)

faultLine = np.load("C:\\Users\\spenc\\Documents\\OkadaPoint\\burstData\\pointGridFaultSample.npy");
faultStrikes = np.load("C:\\Users\\spenc\\Documents\\OkadaPoint\\burstData\\strikeListFault.npy")


#100 nodes per segement (10*10);
#1km x 1km
numNodes = faultLine.shape[0] * faultLine.shape[1];

#defaultDip = math.pi;
defaultDip = math.pi/4;
defaultArea = 1000;
defaultVals = np.zeros(shape = faultLine.shape);

for L in range(len(defaultVals)):
    for W in range(len(defaultVals[0])):
        #area
        #dip
        #strike
        #if(L <numLength or L > numLength + numLengthInner):
        #    defaultVals[L][W][0] = outerArea;
        #    defaultVals[L][W][1] = angleA;
        #    defaultVals[L][W][2] = 0;
        #else:

        segmentNumber = W//50;
        defaultVals[L][W][0] = defaultArea;
        defaultVals[L][W][1] = defaultDip;
        defaultVals[L][W][2] = faultStrikes[segmentNumber];
        
preparedValues = gG.calculatePointValues(faultLine, defaultVals)

rakeVals = np.random.normal(loc = 0, scale = .2, size =numNodes)

slipVals = np.zeros(numNodes)
baseMag = 5;
random.seed(5)
for i in range(len(slipVals)):
    slipVals[i] = .2*random.random();
    #slipVals[i] = 0;


#slipVals = np.load("C:\\Users\\spenc\\Documents\\OkadaPoint\\burstData\\SampleSlip.npy")
test = Simulation(0);
shapedNodeList = np.zeros(shape = (faultLine.shape[0],faultLine.shape[1]),dtype = "object_");

positive = 0;
negative = 0;
for L in range(len(shapedNodeList)):
    for W in range(len(shapedNodeList[0])):
        if(preparedValues[L][W][2] ==1):
            positive = positive + 1;
        elif(preparedValues[L][W][2] ==-1):
            negative = negative + 1;
        else:
            raise Exception("oops");
            
lambd = 3.722*10**9;
mu = 1.2*10**9;

wRange = len(shapedNodeList[0]);

depthAdjust = -10000;

for L in range(len(shapedNodeList)):
    for W in range(len(shapedNodeList[0])):
        faultLine[L][W][2] = faultLine[L][W][2] + depthAdjust;

for L in range(len(shapedNodeList)):
    for W in range(len(shapedNodeList[0])):
        strikeTemp = preparedValues[L][W][0];
        if(preparedValues[L][W][2] == -1): #if negative, reversed fault. Flip Strike angle around
            strikeTemp = strikeTemp + math.pi;
        shapedNodeList[L][W] = OkadaSolutions.OkadaNode(faultLine[L][W][0],faultLine[L][W][1], faultLine[L][W][2],"strike-dip", preparedValues[L][W][3],lambd,mu,preparedValues[L][W][1],strikeTemp,rakeVals[L*wRange + W],L*wRange+W)
        test.NodeManager.addNode(shapedNodeList[L][W])
        
test.NodeManager.secondaryInit(seed,seed,manualInitSlip = True,slipInit = slipVals);
test.preInitPercLoop(interNodeDist,1)

test.NodeManager.mainLoop()

#ax = plt.figure().add_subplot(projection='3d')


baseCFF = np.array(test.NodeManager.CFF);


seedVal = 1063;
cells = np.load("C:\\Users\\spenc\\Documents\\OkadaPoint\\burstData\\" + str(seedVal)+"_0.npy");

numIterations = cells.shape[0];
maxCells = cells.shape[1]

xCoord = 0;
yCoord = 1;
zCoord = 2;

sampleNum = 3;


maxX = -1;
minX = 1000000;

maxZ = -1;
minZ = 1000000;

maxY = -1;
minY = 1000000;

finalIt = 9;

for i in range(maxCells):
    if(cells[finalIt][i] == -1):
        print("NumCells "+ str(i+1));
        break;
    coord = getCoord3D(cells[finalIt][i]);
    if(coord[xCoord] < minX):
        minX = coord[xCoord];
    if(coord[xCoord] > maxX):
        maxX = coord[xCoord];
    if(coord[yCoord] < minY):
        minY = coord[yCoord];
    if(coord[yCoord] > maxY):
        maxY = coord[yCoord];
        
    if(coord[zCoord] < minZ):
        minZ = coord[zCoord];
    if(coord[zCoord] > maxZ):
        maxZ = coord[zCoord];
        
injectionGrid0 = np.zeros(shape = (maxX -minX +1,maxY -minY +1,maxZ -minZ +1));
injectionGrid1 = np.zeros(shape = (maxX -minX +1,maxY -minY +1,maxZ -minZ +1));
injectionGrid2 = np.zeros(shape = (maxX -minX +1,maxY -minY +1,maxZ -minZ +1));
injectionGrid3 = np.zeros(shape = (maxX -minX +1,maxY -minY +1,maxZ -minZ +1));
iterationNum =0;

itr0 = 0;
itr1 = 1;
itr2 = 4;
itr3 = maxCells -1;
baseInjectionSite = ''
for i in range(maxCells):
    if(cells[0][i] == -1):
        print("NumCells "+ str(i+1));
        itr0 = i-1;
        break;
    coord = getCoord3D(cells[0][i]);
    injectionGrid0[coord[xCoord]-minX][coord[yCoord] - minY][coord[zCoord]-minZ] = 1;
for i in range(maxCells):
    if(cells[3][i] == -1):
        print("NumCells "+ str(i+1));
        itr1 = i-1;
        break;
    coord = getCoord3D(cells[3][i]);
    injectionGrid1[coord[xCoord]-minX][coord[yCoord] - minY][coord[zCoord]-minZ] = 1;

for i in range(maxCells):
    if(cells[6][i] == -1):
        print("NumCells "+ str(i+1));
        itr2 = i-1;
        break;
    coord = getCoord3D(cells[6][i]);
    injectionGrid2[coord[xCoord]-minX][coord[yCoord] - minY][coord[zCoord]-minZ] = 1;

for i in range(maxCells):
    if(cells[9][i] == -1):
        print("NumCells "+ str(i+1));
        itr3 = i-1;
        break;
    coord = getCoord3D(cells[9][i]);
    injectionGrid3[coord[xCoord]-minX][coord[yCoord] - minY][coord[zCoord]-minZ] = 1;   


coordsToInject = [];

allCells = cells[9];
injectDistX = 9500;
injectDistY = -200;
injectDistZ = -9500;

pressure = 69 * 10**6;

CFF0 = np.zeros(2000);
CFF1 = np.zeros(2000);
CFF2 = np.zeros(2000);
CFF3 = np.zeros(2000);

gn0 = '';
gs0 = '';

gs1 = '';
gn1 = '';

gs2 = '';
gn2 = ''

gs3 = '';
gn3 = '';
test.NodeManager.resizeGreens(maxCells)


loadingFromFile = True;
if(not loadingFromFile):
    for i in range(maxCells):
        coord = getCoord3D(allCells[i]);
        test.addInflationNode(injectDistX + coord[xCoord] *interNodeDist, injectDistY + coord[yCoord] *interNodeDist, injectDistZ + coord[zCoord] *interNodeDist, pressure, i, 0,initial=True);
        if(i ==itr0):
            print("itr0")
            test.NodeManager.updateStress(); #update the stress values with the new contributions from the inflation node.
            test.NodeManager.update_CFF();
            CFF0 = np.array(test.NodeManager.CFF);
            # gs0 = np.array(test.shearGreens);
            # gn0 = np.array(test.normalGreens)
            # np.save("gn0",gn0);
            # np.save("gs0",gs0);
            np.save("cff0_45_tf",CFF0);
            #del gs0;
            #del gn0;
        if(i == itr1):
            print("itr1")
            test.NodeManager.updateStress(); #update the stress values with the new contributions from the inflation node.
            test.NodeManager.update_CFF();
            CFF1 = np.array(test.NodeManager.CFF);
            #gs1 = np.array(test.shearGreens);
            #gn1 = np.array(test.normalGreens)
            #np.save("gn1",gn1);
            #np.save("gs1",gs1);
            np.save("cff1_45_tf",CFF1);
            #del gs1;
            #del gn1;
        if(i == itr2):
            print("itr2")
            test.NodeManager.updateStress(); #update the stress values with the new contributions from the inflation node.
            test.NodeManager.update_CFF();
            CFF2 = np.array(test.NodeManager.CFF)
            #gs2 = np.array(test.shearGreens);
            #gn2 = np.array(test.normalGreens);
            #np.save("gn2",gn2);
            #np.save("gs2",gs2);
            np.save("cff2_45_tf",CFF2);
            #del gs2;
            #del gn2;
        if(i == itr3):
            print("itr3")
            test.NodeManager.updateStress(); #update the stress values with the new contributions from the inflation node.
            test.NodeManager.update_CFF();
            CFF3 = np.array(test.NodeManager.CFF)
            #gs3 = np.array(test.shearGreens);
            #gn3 = np.array(test.normalGreens);
            #np.save("gn3",gn3);
            #np.save("gs3",gs3);
            np.save("cff3_45_tf",CFF3);
            #del gs3;
            #del gn3;

else:
    CFF0 = np.load("cff0_45_tf.npy");
    CFF1 = np.load("cff1_45_tf.npy");
    CFF2 = np.load("cff2_45_tf.npy")
    CFF3 = np.load("cff3_45_tf.npy")


CFFMapBase = np.zeros(shape = (10,200));
CFFMap0 = np.zeros(shape = (10,200));
CFFMap1 = np.zeros(shape = (10,200));
CFFMap2 = np.zeros(shape = (10,200));
CFFMap3 = np.zeros(shape = (10,200));

for i in range(10):
    for j in range(200):
        CFFMapBase[i][j] = baseCFF[i*200 + j];
        
for i in range(10):
    for j in range(200):
        CFFMap0[i][j] = CFF0[i*200 + j];
        
for i in range(10):
    for j in range(200):
        CFFMap1[i][j] = CFF1[i*200 + j];

for i in range(10):
    for j in range(200):
        CFFMap2[i][j] = CFF2[i*200 + j];

for i in range(10):
    for j in range(200):
        CFFMap3[i][j] = CFF3[i*200 + j];

mod = 1;

CFFMap0 = mod*(CFFMap0 - CFFMapBase);
CFFMap1 = mod*(CFFMap1 - CFFMapBase);
CFFMap2 = mod*(CFFMap2 - CFFMapBase);
CFFMap3 = mod*(CFFMap3 - CFFMapBase);
#CFFMapBase.fill(0)

totalMin = CFFMap0.min();
totalMax = CFFMap3.max();
norm = plt.Normalize(totalMin, totalMax)
# facecolors = plt.cm.plasma(norm(cffMap))

coordFields = gG.getRenderFields(faultLine);

f,axarr = plt.subplots(2,2,figsize=(10,8),subplot_kw=dict(projection='3d'));


angleA = 20
angleB = -45
axarr[0,0].view_init(angleA,angleB)
axarr[1,0].view_init(angleA,angleB)
axarr[0,1].view_init(angleA,angleB)
axarr[1,1].view_init(angleA,angleB)

#f.suptitle("Fracking Cycle effects on Columb Failure Function",fontsize = 20);

axarr[0,0].set_title("1st Injection")
axarr[0,1].set_title("4th Injection")
axarr[1,0].set_title("7th Injection")
axarr[1,1].set_title("10th Injection")

#0 to 20k x
#0 to 1k z
for x in range(2):
    for y in range(2):
        axarr[x,y].axes.set_xlim3d(left = 5000, right=15000) 
        axarr[x,y].axes.set_ylim3d(bottom =-350, top=100) 
        axarr[x,y].axes.set_zlim3d(bottom=depthAdjust, top=1000+depthAdjust) 
        axarr[x,y].axes.xaxis.set_major_locator(MaxNLocator(5));
        axarr[x,y].axes.yaxis.set_major_locator(MaxNLocator(5));
        axarr[x,y].axes.zaxis.set_major_locator(MaxNLocator(5));
        axarr[x,y].set_zlabel("Depth [m]");
        axarr[x,y].set_ylabel("[m]");
        axarr[x,y].set_xlabel("[m]");


axarr[0,1].set_zlabel("Depth [m]");
axarr[1,1].set_zlabel("Depth [m]");
axarr[1,0].set_ylabel("[m]");
axarr[1,1].set_ylabel("[m]");
axarr[1,1].set_xlabel("[m]");
axarr[1,0].set_xlabel("[m]");

renderLimitXMin = 60
renderLimitXMax = 140;
axarr[0,0].plot_surface(coordFields[0][0:10,renderLimitXMin:renderLimitXMax],coordFields[1][0:10,renderLimitXMin:renderLimitXMax],coordFields[2][0:10,renderLimitXMin:renderLimitXMax],facecolors = plt.cm.plasma(norm(CFFMap0[0:10,renderLimitXMin:renderLimitXMax])));
axarr[0,1].plot_surface(coordFields[0][0:10,renderLimitXMin:renderLimitXMax],coordFields[1][0:10,renderLimitXMin:renderLimitXMax],coordFields[2][0:10,renderLimitXMin:renderLimitXMax],facecolors = plt.cm.plasma(norm(CFFMap1[0:10,renderLimitXMin:renderLimitXMax])));
axarr[1,0].plot_surface(coordFields[0][0:10,renderLimitXMin:renderLimitXMax],coordFields[1][0:10,renderLimitXMin:renderLimitXMax],coordFields[2][0:10,renderLimitXMin:renderLimitXMax],facecolors = plt.cm.plasma(norm(CFFMap2[0:10,renderLimitXMin:renderLimitXMax])));
axarr[1,1].plot_surface(coordFields[0][0:10,renderLimitXMin:renderLimitXMax],coordFields[1][0:10,renderLimitXMin:renderLimitXMax],coordFields[2][0:10,renderLimitXMin:renderLimitXMax],facecolors = plt.cm.plasma(norm(CFFMap3[0:10,renderLimitXMin:renderLimitXMax])));

voxelCoords = getVoxelCoord(injectionGrid0, injectDistX, injectDistY, injectDistZ, interNodeDist)
voxelColor = "red"
axarr[0,0].voxels(voxelCoords[0],voxelCoords[1],voxelCoords[2],injectionGrid0,facecolor = voxelColor);
axarr[0,1].voxels(voxelCoords[0],voxelCoords[1],voxelCoords[2],injectionGrid1,facecolor = voxelColor);
axarr[1,0].voxels(voxelCoords[0],voxelCoords[1],voxelCoords[2],injectionGrid2,facecolor = voxelColor);
axarr[1,1].voxels(voxelCoords[0],voxelCoords[1],voxelCoords[2],injectionGrid3,facecolor = voxelColor);



# for x in range(2):
#     for y in range(2):
#         axarr[x,y].axes.set_xlim3d(-100, 400)
#         axarr[x,y].axes.set_ylim3d(-10, 10)
#         axarr[x,y].axes.set_zlim3d(0, 100)
 


cbarTicks = np.arange(totalMin,totalMax,totalMax-totalMin/6);
colormap = plt.cm.get_cmap('plasma')
sm = plt.cm.ScalarMappable(cmap=colormap)
sm.set_clim(vmin=totalMin, vmax=totalMax)
cax = f.add_axes([1, .15, .05,.75])
cb = plt.colorbar(sm,cax = cax)
cb.set_label("$Pa$",rotation= 90);
cb.ax.set_title(r"$\Delta CFF$",pad = 10)


#plt.tight_layout();
plt.savefig("InjectionPlotFault.png",dpi=300);
plt.show()


# fig = plt.figure();
# ax = fig.add_subplot(111, projection='3d');
# ax.plot_surface(X=coordFields[0],Y=coordFields[1],Z=coordFields[2]-10000,facecolors = facecolors);
# cbarTicks = np.arange(totalMin,totalMax,totalMax-totalMin/6);
# colormap = plt.cm.get_cmap('plasma')
# sm = plt.cm.ScalarMappable(cmap=colormap)
# sm.set_clim(vmin=totalMin, vmax=totalMax)
# cax = fig.add_axes([1, .15, .05,.75])
# plt.colorbar(sm,cax = cax)
# ax.set_title('Linear Fault Geometry');
# ax.set_xticks([0,5000,10000,15000,20000]);
# ax.set_yticks([]);
# ax.set_ylim3d(-500,500);
# plt.show()