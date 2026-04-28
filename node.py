# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:51:05 2026

@author: spenc
"""
from abc import ABCMeta,abstractmethod
import numpy as np;
import math;
from scipy.spatial.transform import Rotation as rotation;

class Node:
    #self,x,y,z,pointT,area_or_mag,lamVal,muVal,dipAngle,strikeAngle,rake, nodeID
    @abstractmethod
    def __init__(self,nodeX,nodeY,nodeZ,nodeType,scalingParam,lamVal,muVal,dipAngle,strikeAngle,rakeAngle,nodeID):
        self.mapCoordX = nodeX; #coordinates of the node in space
        self.mapCoordY = nodeY;
        self.mapCoordZ = nodeZ;
        
        
        self.scalingparam = scalingParam; #a value tied to the 'size' of the source. Generally used as a stand in for encompassed area.
        
        self.lam = lamVal; #lame paramters
        self.mu = muVal;
        
        self.dip = dipAngle;
        self.strikeAngle = strikeAngle;
        self.normalvector = self.getNormalVector(self.strikeAngle,dipAngle); #every node that represents a fault plane needs to have a normal vector to define the surface.
        
        self.selfstiffness;
        
        self.stressDrop_Characteristic;
        
        
        
    @abstractmethod
    def getStress(self, otherNode): #returns normal and shear stress caused by this node on a different node (othernode)
        pass;
    
    @abstractmethod
    def getStressTensorAt(self,x,y,z):
        pass;
    
    @abstractmethod
    def selfSample(self): #Some calculation of a nodes stress effects on itself
        pass;
    

    def distance(self,otherNode=0,dataInput = False,otherX = False,otherY = False,otherZ = False): #returns a vector of distance between the current node and another, based on the coord system (0,0,-c)
        if(dataInput == False and otherNode == 0):
            raise Exception("Invalid optionset1");
        if(dataInput == True and otherNode != 0):
            raise Exception("Invalid optionset2");
        if(dataInput == False):
            return np.array([otherNode.mapCoordX - self.mapCoordX, otherNode.mapCoordY-self.mapCoordY, (otherNode.mapCoordZ - self.mapCoordZ) + self.c]);
        
        else:
            return np.array([otherX - self.mapCoordX,otherY - self.mapCoordY, (otherZ-self.mapCoordZ)+self.c])
    

    def getNormalVector(self,strike,dip): #strike is measured clockwise from north. 
        #positive Z is upward
        return np.array([math.cos(strike)*math.sin(dip),-math.sin(dip)*math.sin(strike),math.cos(dip)]);
    
    
    def getRakeVector(self,strike,dip,rake,normal): #currently RADIANS ONLY
        #ortho = np.array([math.cos(strike),math.sin(strike),0]); #base orthogonal vector. Rake of 0 basically.
        ortho = np.array([math.sin(strike),math.cos(strike),0]); 
        rotVec = rotation.from_rotvec(normal*rake); #multiply the magnitude of the rotation onto the unit normal vector. This is the axis of rotation.
        return rotVec.apply(ortho);
        
        