# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 01:31:37 2026

@author: spenc
"""
import node;
import OkadaSolutions;
import numpy as np;
import math;

import sympy;
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import brownian;
import os;
import pickle;
import pandas as pd;
from matplotlib.ticker import MaxNLocator
import geometryGen as gG;
import random;

class NodeManager:
    def __init__(self, eventNumber):
        self.eventNumber =eventNumber; #how many times to run the simulation
        self.slipLaw = False; #if true, using slip law formulation. If false, using slowness law
        self.rateandstate = False;
        self.numNodes = 0;
        self.nodeList = [];
        self.idItr=0;
        self.mu0 = .6;
        self.a = .01;
        self.b = .015;
        self.dc =.1;
        self.g = -9.81; #gravitational constant
        self.RateStateTimeFactor = 1; #years to sample for event;
        self.shearGreens=0;
        self.normalGreens=0;
        self.preInjectionShearGreens = 0;
        self.preInjectionNormalGreens = 0;
        self.NodesFixed = False;
        self.slipMagShear=0;
        self.slipMagNormal=0;
        self.baseRate=0;
        self.slipRate=0; #holds the current slip rate
        self.xiState=0;
        self.dCFF=0;
        self.CFF=0;
        self.CFFNodes = 0;
        self.Vc = 0;
        self.changeNormal = 0; #defining the variables. Actually made in the UpdateStressRate() call
        self.changeShear = 0;
        self.shearStressArray = 0;
        self.normalStressArray = 0; 
        self.initNormSlip = 1;
        self.numCFFNodes = 0;
        
        self.selfStiffnessArray = 0;
        self.numFaultNodes = 0;
        self.seismicMomentLogger = 0;
    
        self.initGreensShear = 0;
        self.initGreensNormal = 0;
        
        self.postInit = False;
        
        self.initFaultSlip = [];
        self.gaussianScale = .05;
        
        self.magOutput = np.zeros(eventNumber +1)
        self.eventItr = 0;
        
        self.verbose = True;
        self.dispOutput = 0;
        self.dispX = 0;
        self.dispY = 0;
        self.dispZ = 0;
        self.seisMag = 0;
        
    def addNode(self,nodeInput,extendNodeList = True):
        nodeInput.nodeID=self.idItr;
        #if(self.NodesFixed):
            #raise Exception("Error: Nodes can not be added after secondary init phase.");
        if(self.numNodes==self.idItr and not extendNodeList):
            raise Exception("numNodes exceeded");
        elif (self.numNodes==self.idItr and extendNodeList):
            self.numNodes= self.numNodes+1;
            self.nodeList.append(nodeInput);
            self.idItr = self.idItr+1;
        else:
            self.nodeList.append(nodeInput);
            self.idItr = self.idItr+1;
    def initGreensArrays(self,faultSeed,paramSeed):
        #print("numNodes", self.numNodes);
        #Setting up the greens function solution matricies    
        offDiagonalMult = .0001;
    
        for i in range(self.numNodes):  #every column shows the effects of node i on all others
            print(i);
            for j in range(0,i):
                if(self.nodeList[j].pointType == "inflation" or self.nodeList[j].pointType =="tensile"): #Want these stress types to only effect strike/dip nodes.
                    self.normalGreens[j][i] = 0;
                    self.shearGreens[j][i] = 0;
                else:
                    stress_temp = self.nodeList[i].getStress(self.nodeList[j]); #calculates effect of node i on node j
                    self.normalGreens[j][i] = offDiagonalMult*stress_temp[0];
                    self.shearGreens[j][i] = offDiagonalMult*stress_temp[1];
            
            if(self.nodeList[i].pointType == "inflation" or self.nodeList[i].pointType =="tensile"):
                self.normalGreens[i][i] = 0;
                self.shearGreens[i][i] = 0;
            else:
                selfSample = self.nodeList[i].selfSample(math.sqrt(self.nodeList[i].area),2); #Sampling 4**2 points.
                #self.normalGreens[i][i] = selfSampleMult *selfSample[0];
                self.normalGreens[i][i] = selfSample[0]
                self.shearGreens[i][i] = selfSample[1];
                self.CFFNodes[i] = 1; #adding this index to the list of nodes to perform CFF calcs on.
            
            self.nodeList[i].selfStiffness = abs(selfSample[1]) - self.mu0*(abs(selfSample[0])+self.nodeList[i].rho * self.nodeList[i].c * self.g); #Default stress state of the node by itself
            self.selfStiffnessArray[i] = abs(selfSample[1]) - self.mu0*(abs(selfSample[0])+self.nodeList[i].rho * self.nodeList[i].c * self.g);
            
            if(self.nodeList[i].selfStiffness>0):
                raise ValueError("Node at index: ",i," with value ",self.nodeList[i].selfStiffness, "is greater than zero");
            for j in range(i+1, self.numNodes):
                if(self.nodeList[j].pointType == "inflation" or self.nodeList[j].pointType =="tensile"): #Want these stress types to only effect strike/dip nodes.
                    self.normalGreens[j][i] = 0;
                    self.shearGreens[j][i] = 0;
                else:
                    stress_temp = self.nodeList[i].getStress(self.nodeList[j]);
                    self.normalGreens[j][i] = offDiagonalMult*stress_temp[0];
                    self.shearGreens[j][i] = offDiagonalMult*stress_temp[1];
        self.numCFFNodes = np.count_nonzero(self.CFFNodes);
       
        self.initGreensShear = self.normalGreens.copy();
        self.initGreensNormal= self.shearGreens.copy();
       
        
        temp = np.zeros(self.numNodes);
        for i in range(len(temp)):
            temp[i] = self.nodeList[i].selfStiffness
            
        #prefix = r"C:\\Users\\witch\\Documents\\Okada113\\InitFiles";
        #prefix = r"C:\Users\spenc\Documents\OkadaPoint";
        prefix = "";
        suffix = "_"+str(faultSeed)+"_"+str(paramSeed)+".npy";
        np.save(prefix+"greensShear"+suffix, self.shearGreens);
        np.save(prefix+"greensNormal"+suffix,self.normalGreens);
        np.save(prefix+"CFFNodes"+suffix,self.CFFNodes);
        np.save(prefix+"selfStiffness"+suffix,self.selfStiffnessArray);
        
        #sanity check
        
        t1 = np.load(prefix+"greensShear"+suffix);
        t2 = np.load(prefix+"greensNormal"+suffix);
        t3 = np.load(prefix+"CFFNodes"+suffix);
        t4 = np.load(prefix+"selfStiffness"+suffix);
         
        print(np.array_equal(self.initGreensShear, t1));
        print(np.array_equal(self.initGreensNormal, t2));
        
        
        print(np.array_equal(self.shearGreens, t1));
        print(np.array_equal(self.normalGreens, t2));
        
        print(np.array_equal(self.CFFNodes, t3));
        print(np.array_equal(self.selfStiffnessArray, t4));
        
        
        
        # exit();
        
        #printGreens = True;
        if(self.verbose):
            print("greens arrays");
            print("normal");
            print(self.normalGreens);
            print("shear");
            print(self.shearGreens);
            
    def getCombinedStressTensor(self,x,y,z):
        temp = np.zeros(shape=(3,3));
        for i in range(self.numNodes):
            temp= temp +self.nodeList[i].getStressTensorAt(x,y,z)*self.slipMagShear[i];
            
        return temp;

    def getCombinedDisplacement(self,x,y,z,slipped,slipMags):
        
        temp = np.zeros(3);
        #only accounting for non-water nodes
        for i in range(self.numCFFNodes):
            if(i in slipped):
                temp = temp + self.nodeList[i].getDisplacementAt(x,y,z)*slipMags[i];
            else:
                offDiagonalMult =1
                temp = temp + offDiagonalMult * self.nodeList[i].getDisplacementAt(x,y,z)*slipMags[i]
        return temp;

    def manualSlipInit(self,slipArr):
        #take in Array of size NumNode. Set shear slip on each
        for i in range(self.numFaultNodes): #setting random starting values
            self.slipMagShear[i] = slipArr[i];
            if(self.slipMagShear[i]<0):
                self.slipMagShear[i] = 0;

    def initSlipValues(self):
        for i in range(self.numFaultNodes): #setting random starting values
            if(self.nodeList[i].pointType == "tensile" or self.nodeList[i] =="inflation"):
                self.slipMagShear[i] = 1; #not relevant to the current sim. Aritrary value;
            else:
                #upperbound = 1*random.random();#random initial max slip val;
                #self.slipMagShear[i] 
                mult = .00001;
                self.slipMagShear[i] = mult * 1
                
                if(self.slipMagShear[i]<0):
                    self.slipMagShear[i] = 0;
    def loadGreensFromFile(self,prefix,files,suffix):
        #files = ["greensShear","greensNormal","CFFNodes","selfStiffness"];
        self.shearGreens = np.load(prefix + files[0] + suffix);
        self.normalGreens = np.load(prefix + files[1] + suffix);
        self.CFFNodes = np.load(prefix + files[2] + suffix);
        self.selfStiffnessArray = np.load(prefix + files[3]+suffix);
        self.numCFFNodes = np.count_nonzero(self.CFFNodes);
        #temp = np.load(prefix + files[3] + suffix);
        for i in range(self.numNodes):
            self.nodeList[i].selfStiffness = self.selfStiffnessArray[i];
        if(self.verbose):    
            print("CHECK");
            print(self.shearGreens.shape);
            print(self.normalGreens.shape);
            print(self.CFFNodes.shape);
            print(self.selfStiffnessArray.shape);
            #print(temp.shape);            
    def checkFiles(self,faultSeed,paramSeed):
        #check if All Needed Files exist;
        suffix = "_"+str(faultSeed)+"_"+str(paramSeed)+".npy";
        #IF YOU NEED TO ADD MORE FILES TO LOAD IN, CHECK HERE
        files = ["greensShear","greensNormal","CFFNodes","selfStiffness"];
        #prefix = r"C:\\Users\\witch\\Documents\\Okada113\\InitFiles";
        #prefix = r"C:\\Users\\spenc\\Documents\\OkadaPoint"
        prefix = "";
        #prefix = r"C:\Users\spenc\Documents\OkadaPoint";
        checkStatus = True;
        
        for i in range(len(files)):
           checkStatus = checkStatus and os.path.isfile(prefix + files[i]+suffix);
        #print("chekc")
        if(checkStatus):
            self.loadGreensFromFile(prefix,files,suffix);
        else:
            self.initGreensArrays(faultSeed,paramSeed)
            
        return checkStatus;
    
    def secondaryInit(self,faultSeed, paramSeed,manualInitSlip = False,slipInit = ""): #init variables that required geometry info. Running this locks the geometry in place.
        self.NodesFixed = True; #locks structure from adding more nodes after arrays are made.
        #greensFromFile = True;
        self.shearGreens = np.zeros(shape=(self.numNodes,self.numNodes));
        self.normalGreens = np.zeros(shape = (self.numNodes,self.numNodes));
        self.CFFNodes = np.zeros(self.numNodes);
        self.selfStiffnessArray = np.zeros(self.numNodes);
        greensFromFile = self.checkFiles(faultSeed,paramSeed);

        self.numFaultNodes = self.numNodes;

        self.slipMagShear = np.zeros(self.numNodes); # need full size array for matrix calcs. Non-slip based nodes (tensile + inflation) will just be set to a constant val, 1 prob.
        self.slipMagNormal = np.zeros(self.numNodes);
        
        self.numFaultPlates = (self.numFaultNodes // 10)+1;

        if(not manualInitSlip):    
            self.initSlipValues();
        else:
            self.manualSlipInit(slipInit);
        
        normalUpperBound = .00001
        for i in range(len(self.slipMagNormal)):
            self.slipMagNormal[i] = normalUpperBound*random.random();
        #self.slipMagNormal = np.ones(self.numNodes);
        self.baseRate = np.zeros(self.numCFFNodes);
        self.slipRate = np.zeros(self.numCFFNodes); #holds the current slip rate
        self.xiState = np.zeros(self.numCFFNodes);
        self.dCFF = np.zeros(self.numCFFNodes);
        self.CFF = np.zeros(self.numCFFNodes);
        for i in range(self.numNodes):
            self.baseRate[i] = self.nodeList[i].base_slip_rate;
            self.slipRate[i] = self.baseRate[i];
            self.xiState = self.dc/self.baseRate[i];
            if(self.rateandstate):
                self.calcVc(i);
        
        self.updateStressRate(); #set starting stress rates based on starting values
        #self.slipStartUp();
        self.updateStress(); #set starting stresses
        #self.updateStressRate(); #set starting stress rate of changes
        self.update_dCFF();
        self.update_CFF();
        counter =0;
        if (self.verbose):
            print("Initial cff");
            print(self.CFF);
         
        self.errorCheck();
    #randomize starting slips
    
    
    
        
    #set starting stresses
    #Stress on a node is the sum of every greens function solution multiplied by current slip
    
    def errorCheck(self):
        if(not np.any(self.dCFF>0)):
            print(self.dCFF)
            raise Exception("Error: All dCFF values are negative");
    
    def updateStress(self,percolation = False):#Stress on a node is the sum of every greens function solution multiplied by current slip
        if(percolation):
            self.shearStressArray = np.matmul(self.preInjectionShearGreens,self.slipMagShear[0:self.numCFFNodes]);
            self.normalStressArray = np.matmul(self.preInjectionNormalGreens,self.slipMagNormal[0:self.numCFFNodes]);
        else:
            self.shearStressArray = np.matmul(self.shearGreens,self.slipMagShear);
            self.normalStressArray = np.matmul(self.normalGreens,self.slipMagNormal);
        #self.normalStressArray = np.matmul(self.normalGreens,self.slipMagShear);
        
        
    #multiplying slip rate by greens solutions to get rate of change. Total rate of change on a node is the sum of each interaction on it, so using matrix multiplication. Used for rate of change of CFF   
    def updateStressRate(self):
        self.changeNormal = np.zeros(self.numNodes); # #keeping normal stress constant for now
        self.changeShear = np.matmul(self.shearGreens,self.baseRate);
    
    def checkSlipDirection(self):
        # for i in range(self.numCFFNodes):
        #     if(self.dCFF[i] < 0):
        #         self.slipRate[i] = -self.slipRate[i];
        return 0;
    def slipStartUp(self):
        self.updateStressRate();
        self.checkSlipDirection();
        self.updateStressRate();
        
    
    def update_dCFF(self):
        for i in range(self.numCFFNodes):
            self.dCFF[i] = abs(self.changeShear[i]) - self.getStaticFrictionCoeff(i)*self.changeNormal[i];
            #self.dCFF[i] = (self.changeShear[i] - self.getStaticFrictionCoeff(i)*self.changeNormal[i]);
            
    def update_CFF(self):
        for i in range(self.numCFFNodes):
            #self.CFF[i] = self.shearStressArray[i] - self.getStaticFrictionCoeff(i)*(self.normalStressArray[i] + self.nodeList[i].rho * self.nodeList[i].c * self.g);
            self.CFF[i] = abs(self.shearStressArray[i]) - self.getStaticFrictionCoeff(i)*(self.normalStressArray[i] + self.nodeList[i].rho * self.nodeList[i].c * self.g);
    
    def lowerSlip(self, index): #lower slip on an index
        #This is the slip lower mechanism in the VQ manual commented below. I've been getting some weird behavior sometimes, so I've been alternating betweeen using this and simply zeroing it out, for testing purposes.
        if(self.verbose):
            print("lowering slip on index ",index)
        temp = self.slipMagShear[index];
        if(temp ==0):
            raise Exception("Trying to lower slip on node with Slip 0");
    
        #randomFactor = np.random.normal(loc = 1.5,scale = .3);
        randomFactor = 1
        if(randomFactor <0):
            randomFactor ==0; #should hopefully never happen, but a sanity check
        self.slipMagShear[index] = self.slipMagShear[index]-randomFactor*1/self.nodeList[index].selfStiffness *(self.nodeList[index].stressDrop_characteristic - self.CFF[index]);
        #self.slipMagShear[index]=0;
        #deltaSlip = temp - self.slipMagShear[index];
        if(self.slipMagShear[index] <0):
            self.slipMagShear[index]=0;
        deltaSlip = temp - self.slipMagShear[index];
        # print("slip")
        # print(temp)
        # print("deltaAdjusted");
        # print(deltaSlip)
        # print(self.slipMagShear[index]);
        #exit();
        # print(self.nodeList[index].mu * self.nodeList[index].area * abs(deltaSlip));
        
        # print("moment, before");
        # print(self.seismicMomentLogger)
        self.seismicMomentLogger = self.seismicMomentLogger + self.nodeList[index].mu * self.nodeList[index].area * abs(deltaSlip);
        # print("moment after");
        # print(self.seismicMomentLogger)
    
    def checkCFFThreshold(self): #checks the CFF values to see if any are past the slip threshold. If so, lowers the slip. Does this for all nodes with CFF>0
        #print(self.CFF);
        slippedNodes = set();
        nodeSlipped = False;
        for i in range(self.numCFFNodes):
            #if(self.CFF[i] >= 0 and self.dCFF[i] >0):
            if(self.CFF[i] >= 0):
                if(self.dCFF[i] <0):
                    "Warn: Slip on DCFF < 0 Node";
                if(self.verbose):
                    print("Node slipped at index ",i," with CFF value ",self.CFF[i]);
                nodeSlipped = True;
                if(self.verbose):
                    print("Lowering slip at Node\n");
                slippedNodes.add(i);
        return (nodeSlipped,slippedNodes);
    def reshuffleSlips(self):
        for i in range(len(self.slipMagShear)):
            magVal = 1;
            self.slipMagShear[i] = random.random()*magVal;
            
    def lowerSlipsAndUpdateStress(self, slippedNodes,percolation=False):
        for i in range(self.numCFFNodes):
            if i in slippedNodes:
                self.lowerSlip(i);
        self.updateStress(percolation);
        self.update_CFF();
        
        return self.checkCFFThreshold();
    def aftershockLoop(self,initSetup=False,percolation=False):
        slipped = True;
        self.breakCondition = False;
        aftershockCount= 0;
        self.seismicMomentLogger = 0;
        currentSlipped = 0;
        pastSlipped =  set();
        # print("init?")
        # print(initSetup)
        # print("CFF at start of aftershock Loop")
        # print(self.CFF)
        #print(self.slipMagShear)
        while(slipped): #continues if there's something that slipped;
            res = self.checkCFFThreshold(); #checks if there are any CFF's equal to 0 or above it. If so, it lowers the slip on them. An Earthquake on it.
            slipped = res[0];
            currentSlipped = res[1];
            # print(self.CFF)
            # print(self.slipMagShear)
            # if(len(pastSlipped)!=0):
            #     if(len(currentSlipped.intersection(pastSlipped)) >0):
            #         print("Error: Repeated Slip");
            #         print(currentSlipped.intersection(pastSlipped));
            #         self.reshuffleSlips();
            #         break;
                    #exit();
            if(slipped): #means at least one node has lowered slip now. Need to recalculate stress on everything, and then check again.
                aftershockCount = aftershockCount +1;
                pastSlipped.update(currentSlipped);
                stableState = False;
                #while(not stableState):
                #    stableState = self.lowerSlipsAndUpdateStress(currentSlipped,percolation);
                    
                #if(aftershockCount==10):
                #    raise Exception("Error, repeating in aftershock loops.");
                if(self.verbose and not initSetup):
                    print("Nodes slipped, updating values and rechecking");
                #print(self.CFF[982]);
                #print(res)
                #self.updateStress(percolation); #recalculates stress values on every node based on current slips;
                #self.update_CFF(); #update CFF based on new normal and shear stress values
                res = self.lowerSlipsAndUpdateStress(currentSlipped,percolation)
                #print(res)
                #exit()
                #print(self.CFF[982]);
               # self.breakCondition=True;
            else:
                #print("No Nodes Slipped\n----------------------------------")
                # if(self.seismicMomentLogger>0 and not initSetup):
                #     print("Seismic Moment:" , self.seismicMomentLogger);
                #     print("Seis. Mag: ", 2/3 * math.log10(self.seismicMomentLogger) - 10.7);
                pastSlipped.update(currentSlipped);
                if(not initSetup):
                    if(self.verbose):
                        print("Aftershock Count ",aftershockCount)
                        #print("Seismic Moment:" , self.seismicMomentLogger);
                        if(self.seismicMomentLogger == 0):
                            print("err 0")
                            continue;
                            print("Seis. Mag: ", 2/3 * math.log10(self.seismicMomentLogger) - 10.7);
                    if(self.seismicMomentLogger==0):
                        self.magOutput[self.eventItr] =0;
                    else:
                        self.magOutput[self.eventItr] = 2/3 * math.log10(self.seismicMomentLogger) - 10.7
                    self.eventItr = self.eventItr +1;
                    return pastSlipped;
                    # with open("testOutput.txt", "a") as f:
                    #     f.write(str(2/3 * math.log10(self.seismicMomentLogger) - 10.7)+"\n");
                    
                if(self.postInit and self.breakCondition):
                    return pastSlipped;
                    #break;
        #print("Final CFF", self.CFF);
                
    
    #calculates values for a given node ID 
    def getStaticFrictionCoeff(self,index=-1):
        if(self.rateandstate):
            if(index == -1):
                raise Exception("Requires node ID");
            return self.mu0 + self.a*math.log(self.slipRate[index]/self.baseRate[index]) + self.b*math.log(self.baseRate[index]*self.xiState[index]/self.dc);
        else:
            return self.mu0;
    
    def dXidT(self,index = -1):
        if(self.rateandstate and self.slipLaw):
            if(index == -1):
                raise Exception("Requires node ID");
            return -(self.slipRate[index]*self.xiState[index]/self.dc)*math.log(self.slipRate[index]*self.xiState[index]/self.dc);
        elif(self.rateandstate and not self.slipLaw):
            if(index == -1):
                raise Exception("Requires node ID");
            return 1- (self.slipRate[index]*self.xiState[index]/self.dc);
        else:
            return 0;
    
    def calcVc(self,index = -1):
        self.Vc[index] = math.exp(-(self.mu0 + self.a * math.log(self.baseRate[index]) - self.b*math.log(self.baseRate[index]/self.dc))/self.a);
    
    def updateSlipRate(self,index =-1):
        if(self.rateandstate):
            if(index == -1):
                raise Exception("Requires node ID");
            self.slipRate[index] = self.Vc[index]* math.pow(self.xiState[index],(-self.b/self.a))*math.exp(self.shearStressArray[index]/(self.normalStressArray[index]*self.a));
    def resizeGreens(self,increase):
        self.shearGreens = np.pad(self.shearGreens,((0,increase),(0,increase)),mode='constant',constant_values=0);
        self.normalGreens = np.pad(self.normalGreens,((0,increase),(0,increase)),mode='constant',constant_values=0);
        self.slipMagShear = np.pad(self.slipMagShear,(0,increase),mode='constant', constant_values=0);
        self.slipMagNormal = np.pad(self.slipMagNormal,(0,increase),mode='constant', constant_values=0);  
    
    def resetSlips(self):
        self.shearGreens = self.initGreensShear.copy();
        self.normalGreens = self.initGreensNormal.copy();
        self.numNodes = self.numFaultNodes;
        self.idItr=self.numNodes;
        self.nodeList = self.nodeList[0:self.numNodes];
        self.slipMagShear = np.zeros(self.numNodes);
        self.slipMagNormal = np.ones(self.numNodes);
        
        self.initSlipValues();
        self.updateStress();
        self.update_CFF();
        
        #print(self.shearGreens.shape);
        #print(self.normalGreens.shape);
        #print(self.numNodes);
        
        
        self.aftershockLoop(initSetup=True);
    
    def timeAdvancement(self):
        if(self.rateandstate):
            print("temp");
            #This is a very work in progress section. Time advancement isn't linear unlike the normal model, so there's a degree of uncertaintity about when the next event is. Currently just sampling over a repeating time interval;
            
        else:
            #check cff values and divide by rate of change of cff
            #find lowest time to advance, advance forward
            #ignore ones with negative dCFF for time selection criteria
            #dCFF is constant, so need to call a refresh on it.
            #print("dCFF",self.dCFF);
            mindex = -1;
            minVal = float('inf');
            for i in range(len(self.dCFF)):
                if(self.CFF[i]>=0):
                    print("ERROR FEED");
                    #print(self.CFF);
                    #print(self.slipMagShear)
                    raise Exception("CFF should not be greater than 0 in this stage");
                if(self.dCFF[i] >=0):
                    time = -self.CFF[i]/self.dCFF[i];
                    #print(time);
                    if (time<minVal):
                        minVal = time                                                             ;
                        mindex = i;
            #print("times:", -self.CFF/self.dCFF);
            #print(self.dCFF)
            #fix non-ZeroCFF
            timeMult = 1.000001; #make it slightly longer so it gets fully to zero
            if(self.verbose):
                print("mindex:",mindex);
                print("Advancing forward in time:", timeMult*minVal/(86400*365.25), " years."); #not quite hitting 0 all the time exactly. I think it's a numerical precision issue.
            for i in range(self.numCFFNodes):
                #if(self.dCFF[i]>=0):
                self.slipMagShear[i] = self.slipMagShear[i] + timeMult*minVal * self.slipRate[i];
            self.updateStress();
            self.update_CFF();
            #print("here");
            # for i in range(self.numCFFNodes):
            #     #if(self.dCFF[i]>=0):
            #     if(self.CFF[i]>0):
            #         print(i)
            #         print(self.CFF[i]," ",self.dCFF[i])
            #exit();
            #print(self.CFF)
            if(self.verbose):
                print("Nodes to Slip")
                for i in range(self.numNodes):
                    if(self.CFF[i]>0):
                        print(i," ",self.CFF[i]);
            #print("time advanced:",self.CFF)

    def mainLoop(self):
        firstPass = True;
        
        loopCondition = True; 
        loopitr = 0;
        if(self.verbose):
            print("Beginning main simulation loop");
        while(loopCondition):
            if(self.verbose):
                print("CFF check");
            if(firstPass== True):
                self.aftershockLoop(initSetup=True); #check for CFF greater than or equal to 0; starts lowering slips and updating.
                firstPass = False;
            else:
                self.aftershockLoop();
            #if(loopitr==self.eventNumber and not self.postInit):
            if(loopitr==self.eventNumber):
                if(self.verbose):    
                    print("Done!")
                break;
            if(self.verbose):
                print("\nSimulation Iteration ",loopitr+1," completed. Checking time advancement.");
            #print("Current CFF:",self.CFF);
            #print("Current Slip",self.slipMagShear);
            self.timeAdvancement(); #find time required to advance forward, and start interating values.
            loopitr = loopitr+1;
            if(loopitr==self.eventNumber and not self.postInit):
                self.aftershockLoop()
                if(self.verbose):    
                    print("Done!")
                break;
        self.preInjectionNormalGreens = self.normalGreens.copy();
        self.preInjectionShearGreens = self.shearGreens.copy();