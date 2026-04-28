# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:25:54 2024

@author: witch
"""

import numpy as np;
import math;
import time;
import brownian;
import matplotlib.pyplot as plt;
import random;
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap



horizontal = np.asarray([0,1,0]);
strikeZero = [1,0,0];

#to get dip, get angle between horizontal and projected angle

def getStrikeRight(testVector):
    zeroVector = [1,0,0];
    testVector[2]=0; #project onto x/y
    neg= -1;
    if(testVector[1]>0):
        neg=1;        
    theta = math.acos( np.dot(testVector,zeroVector)/(np.linalg.norm(testVector)));
    return neg*theta;

def getStrikeLeft(testVector):
    zeroVector = [-1,0,0];
    testVector[2]=0; #project onto x/y
    neg= 1;
    if(testVector[1]>0):
        neg=-1;        
    theta = math.acos( np.dot(testVector,zeroVector)/(np.linalg.norm(testVector)));
    return neg*theta;

def getStrikeVector(testVector):
    testVector[2]= 0;
    theta = math.acos(np.dot(testVector,strikeZero)/(np.linalg.norm(testVector)));
    neg = 1;
    if(testVector[1]<0):
        neg=-1;        
    return neg* theta;


def getDipVector(testVector):
    #unitTest = np.divide(testVector,np.linalg.norm(testVector));
    testVector[0]=0;
    theta = math.acos( np.dot(testVector,horizontal)/(np.linalg.norm(testVector)));
    return theta;

def checkDir(dipAngle):
    if(dipAngle > math.pi):
        raise Exception("oopsie doopsie");
    if(dipAngle>math.pi/2):
        return [math.pi-dipAngle,-1];
    else:
        return [dipAngle,1];

def getDipAngle(centralPoint,lowerPoint,upperPoint):
    t1="";
    t2="";
    if(upperPoint is None and lowerPoint is None):
        raise Exception("ERROR. NULL upper and lower point");
    if(upperPoint is None):
        distLower = np.subtract(centralPoint,lowerPoint);
        return checkDir(getDipVector(distLower));
    else:
        distUpper = np.subtract(upperPoint,centralPoint);
        t2 = getDipVector(distUpper);
    if(lowerPoint is None):
        distUpper = np.subtract(upperPoint,centralPoint);
        return checkDir(getDipVector(distUpper));
    else:
        distLower = np.subtract(centralPoint,lowerPoint);
        t1 = getDipVector(distLower);
    return checkDir((t1 + t2)/2);

def getStrikeAngle(centralPoint,leftPoint,rightPoint):
    
    if(leftPoint is None and rightPoint is None):
        raise Exception("null left and right");
    if(leftPoint is None):
        distRight = np.subtract(rightPoint,centralPoint);
        #print(distRight);
        return getStrikeVector(distRight);
    elif (rightPoint is None):
        distLeft = np.subtract(centralPoint,leftPoint);
        #print(distLeft);
        return getStrikeVector(distLeft);
    else:
        distLeft = np.subtract(centralPoint,leftPoint);
        distRight = np.subtract(rightPoint,centralPoint);
        t1 = getStrikeVector(distLeft);
        t2 = getStrikeVector(distRight);
        return (t1+t2)/2
    

def transformArea(baseArea,dipDif,strikeDif):
    #takes a base area assumed for a region with a given dip and strike. Get difference between them and the real point. Transform area.
    maxMul = 10; 
    if(dipDif==0):
        return baseArea/math.cos(strikeDif);
    temp = abs(baseArea/(math.sin(dipDif)*math.cos(strikeDif)));
    if(maxMul * baseArea < temp):
        return maxMul * baseArea;
    else:
        return abs(baseArea/(math.sin(dipDif)*math.cos(strikeDif)));
    #return abs(baseArea);

def calculatePointValues(pointArray, defaultVals):
    #take in an array of coordinates for points on fault. XYZ.
    #Points Wide * total Points down across 3 segements;
    #roughly 3N * W * 3
    #edge points get mirrored strike/dip. They are what they appear to be from the one direction
    #eg, 45 angle to 0 on one side is 45 total
    #starts at top left
    
    #defaultVals is a L*W*3 array with default values of the value in that region
    #0 = area
    #1 = dip
    #2 = strike
    #returning important info
    length = len(pointArray);
    width = len(pointArray[0]);
    if(length == 0 or width ==0):
        raise Exception("invalid input")
    outArray = np.zeros(shape =(length,width,4));
    #0 - Strike
    #1 - Dip
    #2 - Direction (is the fault reversed on this section)
    #3 - area
    for Li in range(length):
        for Wi in range(len(pointArray[0])):
            #for a particular point, check if it's an edge point
            dipEdge = False;
            strikeEdge = False;
            if(Li ==0 or Li == length -1):
                dipEdge = True;
            if(Wi == 0 or Wi == width -1):
                strikeEdge = True;
            if(Li !=0 and Li != length-1):
                #not an edge, normal operation
                #getDipAngle(centralPoint, lowerPoint, upperPoint)
                temp = getDipAngle(pointArray[Li][Wi],pointArray[Li-1][Wi],pointArray[Li+1][Wi])
                outArray[Li,Wi,1] = temp[0];
                outArray[Li,Wi,2] = temp[1];
            else:
                if(Li == 0):
                    temp = getDipAngle(pointArray[Li][Wi],None,pointArray[Li+1][Wi]);
                    outArray[Li,Wi,1] = temp[0];
                    outArray[Li,Wi,2] = temp[1];
                if(Li == length -1):
                    temp= getDipAngle(pointArray[Li][Wi],pointArray[Li-1][Wi],None);
                    outArray[Li,Wi,1] = temp[0];
                    outArray[Li,Wi,2] = temp[1];
            if(Wi != 0 and Wi != width -1):
                #getStrikeAngle(centralPoint,leftPoint,rightPoint):
                outArray[Li][Wi][0]= getStrikeAngle(pointArray[Li][Wi],pointArray[Li][Wi-1],pointArray[Li][Wi+1]);
            else:
                if(Wi == 0):
                    outArray[Li][Wi][0] = getStrikeAngle(pointArray[Li][Wi],None,pointArray[Li][Wi+1]);
                if(Wi == width-1):
                    outArray[Li][Wi][0] = getStrikeAngle(pointArray[Li][Wi],pointArray[Li][Wi-1],None);
            outArray[Li][Wi][3] = transformArea(defaultVals[Li][Wi][0],abs(defaultVals[Li][Wi][1] -outArray[Li][Wi][1]),abs(defaultVals[Li][Wi][2] -outArray[Li][Wi][0]))
           # outArray[Li,Wi,0] = getStrikeAngle(pointArray[Li][Wi], pointArray[Li][Wi-1], pointArray[Li][Wi+1]);
           # outArray[Li,Wi,1] = getDipAngle(pointArray[Li][Wi],pointArray[Li+1][Wi],pointArray[Li-1][Wi]);
    return outArray;

def rescaleEdges(grid,both=True,horizontal = False,softness = .1):
    #rescale edge of grid so it goes to zero alteration. By default 0,0 is always 0
    if(softness > .5):
        raise "Error";
    tempGrid=0;
    if(horizontal):
        tempGrid = grid.transpose();
    else:
        tempGrid = grid;
    gridY = len(tempGrid)
    gridX = len(tempGrid[0]);
    lim = math.floor(gridY * softness);
    for i in range(gridX):
        for y in range(lim):
            tempGrid[y][i] = tempGrid[y][i]* y/lim;
    
    if(both):
        tempGrid = np.flip(tempGrid,axis = 0);
        for i in range(gridX):
            for y in range(lim):
                tempGrid[y][i] = tempGrid[y][i]* y/lim;
        grid = np.flip(grid,axis = 0);
    grid = tempGrid.transpose();
    return grid;

def rotationRx(vector, angle): #rotation around x Axis
    angleAdjust = -(math.pi/2 - angle);
    rotMatrix = np.array([[1,0,0],[0,math.cos(angleAdjust),-math.sin(angleAdjust)],[0,math.sin(angleAdjust),math.cos(angleAdjust)]]);
    return np.matmul(rotMatrix,vector);


def rotationRz(vector,angle):
    rotMatrix = np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]]);
    return np.matmul(rotMatrix,vector);

def planeGen(xOrg,yOrg,zOrg,topbot,W,L,numWidth,numLength,angle,insideEdge=True,isBrownian =True):
    genSize = numWidth;
    widthLarger = True;
    if(numLength>numWidth):
        genSize = numLength;
        widthLarger = False;

    fieldShape = True;
    #angle from horizontal, from origin point.=
    #topbot is if origin is in bottom left of plane or top left
    #width horiztonal, length height on plane
    pointList = np.zeros(shape = (numLength*numWidth,3));
    #print(pointList.shape);
    lengthPoints = np.zeros(numLength);
    widthPoints = np.zeros(numWidth);
    for i in range(numLength):
        lengthPoints[i] = L/numLength *i + L/(2*numLength);
    for i in range(numWidth):
        widthPoints[i] = W/numWidth * i + W/(2*numWidth);
        
    #print(lengthPoints);
    #print(widthPoints);
    if(topbot != "bot" and topbot !="top"):
        raise Exception("badtopbot");
    #X coord is down length of fault
    #Y coord is from rotation from dip
    #Z is up and down. Vertical part of fault if no dip.
    brownField = brownian.brownian_surface(R=2,N=genSize*2, H=0.55);
    subsetBrown = np.zeros(shape = (numLength,numWidth));
    for w in range(numWidth):
        for l in range(numLength):
            subsetBrown[l][w] = brownField[l][w];
    
    # print()
    # print("BrownField")
    # print(brownField.shape)
    # print("Subset")
    # print(subsetBrown.shape)
    # print(subsetBrown[0][0]);
    # print(numLength);
    # print(numWidth)
    
    magnitude = .9;
    #brownField = rescaleEdges(brownField,both = insideEdge,softness = .1)
    subsetBrown = rescaleEdges(subsetBrown,both=insideEdge,softness = .3);
    #print("brown")
    #print(brownField)
    #isBrownian=True;
    itr = 0;
    outputGrid = np.zeros(shape = (len(lengthPoints),len(widthPoints),3))
    # for length in range( len(lengthPoints)):
    #     for width in range(len(widthPoints)):
    #         #print(width)
    #         pointList[itr][0] = widthPoints[width];
    #         if(isBrownian):
    #             pointList[itr][1] = magnitude*subsetBrown[width][length];
    #         else:
    #             pointList[itr][1]=0
    #         pointList[itr][2] = lengthPoints[length];
    #         itr = itr + 1;    
    for length in range( len(lengthPoints)):
        for width in range(len(widthPoints)):
            #print(width)
            outputGrid[length][width][0] = widthPoints[width]
            if(isBrownian):
                #outputGrid[length][width][1] = magnitude*subsetBrown[width][length];
                outputGrid[length][width][1] = magnitude*subsetBrown[length][width];
            else:
                outputGrid[length][width][1]=0
            outputGrid[length][width][2] = lengthPoints[length];
    

    for length in range( len(lengthPoints)):
        for width in range(len(widthPoints)):
            vect = outputGrid[length][width];
            temp = rotationRx(vect,angle);
            if (topbot == "bot"):
                temp[0] = temp[0] + xOrg;
                temp[1] = temp[1] + yOrg;
                temp[2] = temp[2] + zOrg;
            else:
                temp[0] = temp[0] + xOrg;
                temp[1] = yOrg - temp[1];
                temp[2] = zOrg - temp[2];
            outputGrid[length][width] = temp;
            

    
    return outputGrid;


def planeGenHorizontal(xOrg,yOrg,zOrg,topbot,W,L,numWidth,numLength,dipAngle,strikeAngle,insideEdge =True,isBrownian =True,brownianMag =1,HVal =.55 ):
    #dampen: False, Left, Right, Both
    genSize = numWidth;
    widthLarger = True;
    if(numLength>numWidth):
        genSize = numLength;
        widthLarger = False;

    fieldShape = True;
    #angle from horizontal, from origin point.=
    #topbot is if origin is in bottom left of plane or top left
    #width horiztonal, length height on plane
    pointList = np.zeros(shape = (numLength*numWidth,3));
    #print(pointList.shape);
    lengthPoints = np.zeros(numLength);
    widthPoints = np.zeros(numWidth);
    for i in range(numLength):
        lengthPoints[i] = L/numLength *i + L/(2*numLength);
    for i in range(numWidth):
        widthPoints[i] = W/numWidth * i + W/(2*numWidth);
        
    #print(lengthPoints);
    #print(widthPoints);
    if(topbot != "bot" and topbot !="top"):
        raise Exception("badtopbot");
    #X coord is down length of fault
    #Y coord is from rotation from dip
    #Z is up and down. Vertical part of fault if no dip.
    brownField = brownian.brownian_surface(R=2,N=genSize*2, H=HVal);
    subsetBrown = np.zeros(shape = (numLength,numWidth));
    for w in range(numWidth):
        for l in range(numLength):
            subsetBrown[l][w] = brownField[l][w];
    
    # print()
    # print("BrownField")
    # print(brownField.shape)
    # print("Subset")
    # print(subsetBrown.shape)
    # print(subsetBrown[0][0]);
    # print(numLength);
    # print(numWidth)
    
    magnitude = brownianMag;
    #brownField = rescaleEdges(brownField,both = insideEdge,softness = .1)
    subsetBrown = rescaleEdges(subsetBrown,both=insideEdge,horizontal=True,softness = .2);
    #print("brown")
    #print(brownField)
    #isBrownian=True;
    itr = 0;
    outputGrid = np.zeros(shape = (len(lengthPoints),len(widthPoints),3))
    # for length in range( len(lengthPoints)):
    #     for width in range(len(widthPoints)):
    #         #print(width)
    #         pointList[itr][0] = widthPoints[width];
    #         if(isBrownian):
    #             pointList[itr][1] = magnitude*subsetBrown[width][length];
    #         else:
    #             pointList[itr][1]=0
    #         pointList[itr][2] = lengthPoints[length];
    #         itr = itr + 1;    
    for length in range( len(lengthPoints)):
        for width in range(len(widthPoints)):
            #print(width)
            outputGrid[length][width][0] = widthPoints[width]
            if(isBrownian):
                #outputGrid[length][width][1] = magnitude*subsetBrown[width][length];
                outputGrid[length][width][1] = magnitude*subsetBrown[length][width];
            else:
                outputGrid[length][width][1]=0
            outputGrid[length][width][2] = lengthPoints[length];
    

    
    for length in range( len(lengthPoints)):
        for width in range(len(widthPoints)):
            vect = outputGrid[length][width];
            temp = rotationRz(rotationRx(vect,dipAngle),strikeAngle);
            #temp = rotationRx(vect,dipAngle);
            if (topbot == "bot"):
                temp[0] = temp[0] + xOrg;
                temp[1] = temp[1] + yOrg;
                temp[2] = temp[2] + zOrg;
            else:
                temp[0] = temp[0] + xOrg;
                temp[1] = yOrg - temp[1];
                temp[2] = zOrg - temp[2];
            outputGrid[length][width] = temp;
            

    
    return outputGrid;

def getRenderFields(pointGrid):
    #pointGrid is a LxWx3 grid
    #returning 3 LxW grids containing each coordinate val
    numLength = pointGrid.shape[0];
    numWidth = pointGrid.shape[1]
    xField = np.zeros(shape = (numLength,numWidth));
    yField = np.zeros(shape = (numLength,numWidth));
    zField = np.zeros(shape = (numLength,numWidth));
    #print("renderField")
    #print(xField.shape);
    #print(yField.shape);
    #print(zField.shape);
    for i in range(numLength):
        for j in range(numWidth):
            xField[i][j] = pointGrid[i][j][0];
            yField[i][j] = pointGrid[i][j][1];
            zField[i][j] = pointGrid[i][j][2];
    return (xField,yField,zField);

def renderPointGrid(pointGrid,saveFig = False,elev = 20,azim =-20,xlim =-1,ylim=-1,zlim=-1,cmap =-1):
    #pointGrid is a LxWx3 grid
    coordFields = getRenderFields(pointGrid);
    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d');
    if(cmap != -1):
        ax.plot_surface(coordFields[0],coordFields[1],coordFields[2],cmap=cmap);
    else:
        ax.plot_surface(coordFields[0],coordFields[1],coordFields[2]);
    ax.view_init(elev,azim)
    if(xlim != -1):
        ax.set_xlim3d(xlim[0],xlim[1]);
    if(ylim != -1):    
        ax.set_ylim3d(ylim[0],ylim[1]);
    if(zlim != -1):
        ax.set_zlim3d(zlim[0],zlim[1]);
    plt.tight_layout();
    ts = time.time()
    plt.show()
    if(saveFig):
        plt.savefig("testFault_"+str(ts)+".png",dpi=1000);

def generateFaultGeometry():
    Rval = 2;
    galleryDepth = -3000;
    galleryAdjust = 0;
    faultLength = 50;
    #angleA = .559;
    angleA=.7
    angleB = 1.097;
    numWidth  = 100;
    widthLength = 100;
    numLength = 20;
    numLengthInner = numLength//2;
    numWidth = 20;
    numNodes = (2*numLength + numLengthInner)*numWidth;
    # brownField = brownian.brownian_surface(R=2,N=100, H=0.85);
    # print(brownField)
    # print(rescaleEdges(brownField))
    
    #deltaX/Cos(angle)
    innerLength = 6.7285;
    
    brownian = False;
    p1 = planeGen(0,0,galleryDepth + galleryAdjust,"bot",widthLength,innerLength,numWidth,numLengthInner,angleB,insideEdge = True,isBrownian =brownian);
    p2 = planeGen(0,0,galleryDepth + galleryAdjust,"top",widthLength,innerLength*2,numWidth,numLength,angleA,insideEdge = False,isBrownian =brownian);
    p3 = planeGen(0,p1[numLengthInner-1][0][1],p1[numLengthInner-1][0][2],"bot",widthLength,innerLength*2,numWidth,numLength,angleA,insideEdge = False,isBrownian =brownian);
    combo = np.concatenate((np.flip(p2,axis = 0), p1, p3));
    return combo;





def generateHoriztonalTestFault():
    x0 = 0;
    y0 = 0;
    z0 = 0;
    W = 1000;
    L = 1000;
    numWidth = 50;
    numLength = 50;
    dipAngle = math.pi/2;
    strikeAngle = 0;
    p1 = planeGenHorizontal(x0, y0, z0, "bot", W, L, numWidth, numLength, dipAngle, strikeAngle,isBrownian=True )
    print(p1[0][numLength-1]);
    return p1;

def randomAngle():
    strikeMax = math.pi/16
    return strikeMax *(random.random() -.5); #-.5 to .5

def gaussianRandom(currentStrike):
    averageStrike = 0;
    delta = averageStrike - currentStrike; #poscurrent = negative mean 
    return np.random.normal(0,.025);
def generateHoriztontalSegment(x0,y0,z0,strike,brownian):
    W = 1000;
    L = 1000;
    numWidth = 10;
    numLength = 10;
    dipAngle = math.pi/2;
    plane = planeGenHorizontal(x0, y0, z0, "bot", W, L, numWidth, numLength, dipAngle, strike,isBrownian=brownian,brownianMag=15,HVal = .8);
    return plane, plane[0,numLength-1];
    
def generateHorizontalChain():
    x0 = 0;
    y0 = 0;
    z0 = 0;
    #40 has potential
    seedVal = 31;
    random.seed(seedVal);
    np.random.seed(seedVal);
    maxStrike = math.pi/4;
    numSegments = 20;
    #strikes = np.zeros(numSegments)
    currentX = x0;
    currentY = y0;
    currentZ = z0;
    currentStrike =0;
    W = 1000;
    numWidth = 50;
    finalPoints = [];
    #strikes = [0,randomAngle(),randomAngle()
    #for i in range(numSegments):
    #    strikes[i] = randomAngle();
            
    #print(strikes)
    for i in range(numSegments):
        plane,finalPoint = generateHoriztontalSegment(currentX, currentY, currentZ, currentStrike)
        if(i == 0):
            finalPoints = plane;
        else:
            #print(finalPoints.shape);
            #print(plane.shape)
            finalPoints = np.concatenate((finalPoints,plane),axis =1);
        currentX = x0 + finalPoint[0] + W/numWidth *math.cos(currentStrike);
        #positive strike means -y
        currentY = y0 + finalPoint[1] - W/numWidth *math.sin(currentStrike);
        currentStrike = gaussianRandom(currentStrike);
        #print("strike: ",currentStrike)
        
    return finalPoints;
if __name__=="__main__":
    print("test")
    #Assorted Testing Code that can be uncommented.
    # p1 = [0,-1,-1];
    # p2 = [0,0,0];
    # p3 = [0,1,1];
    #L = 5;
    
    #p1 = [0,1,-1];
    #p2 = [0,0,0];
    #p3 = [0,-1,1];
    
    #print(checkDir(3*math.pi/4))
    
    #getDipAngle(centralPoint, lowerPoint, upperPoint)
   # out = getDipAngle(p2,p3,p1);
    #print(out);
    #print( math.degrees(out[0]));
    
    
    #temp = generateFaultGeometry();
    #temp  = generateHoriztonalTestFault();
    
    #p1 = [-1,1,0];
    #p2 = [-.5,0,0];
    #p3 = [0,-1,0];
    
    #verOut = calculateDipAndStrike(testArray)
    
    #getStrikeAngle(p2,p1,p3)