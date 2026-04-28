# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 19:11:41 2023

@author: Spence
"""
import numpy as np;
import math;
from scipy.spatial.transform import Rotation as rotation;
from node import Node;

class OkadaNode(Node):
    # pointType = "null";
    # c = 0;
    # delta = 0;
    # M0 = 0;
    # lam = 0;
    # mu = 0;
    
    # mapCoordX =0;
    # mapCoordY =0;
    # mapCoordZ =0;
    
    # isDisplacement = False;
    
    def __init__(self,x,y,z,pointT,area_or_mag,lamVal,muVal,dipAngle,strikeAngle,rake, nodeID): #Z is positive above ground and negative below
        self.mapCoordX =x;
        self.mapCoordY = y;
        self.mapCoordZ = z;
        self.c = z; 
        self.lam = lamVal;
        self.mu = muVal;
        self.poisson = self.lam/(2*(self.lam+self.mu));
        #self.M0 = area*self.mu;
        self.area = area_or_mag; #currently in meters squared. 
        self.nodeID = -1;
        self.isDisplacement = False;
        #self.stressDropFactor = -1;
        self.stressDropFactor = .3;
        self.Mchar =0; #
        #self.meanSlip = (10**(3/2 * (self.Mchar + 10.7)))/(10**7 * self.mu*self.area);
        self.meanSlip = 0;
        self.L = 0;
        self.stressDrop_characteristic =0;
        #self.meanSlip = 50;
        
        if(pointT =="inflation"):
            self.M0=area_or_mag;
        else:
            self.Mchar = 4 + math.log10(self.area/(1000000)) + self.stressDropFactor;
            self.meanSlip = 10**(3/2 * (self.Mchar + 6))/(self.mu * self.area);
            self.M0 = self.mu*self.area*self.meanSlip;
            L = math.sqrt(self.area/(1000000));
            self.stressDrop_characteristic = -(2*self.mu *self.meanSlip)/((1-self.poisson)*math.pi*math.sqrt(2*L**2))*(2-self.poisson);
        #print("M0",self.M0);
        #print("MeanSlip",self.meanSlip)
        #assuming L and W are equal for below calc, currently. See VQ manual for original formula. If L=W, then L=sqrt(Area) -> R = sqrt(2*area)
        #self.stressDrop_characteristic = -(2*self.mu *self.meanSlip)/((1-self.poisson)*math.pi*math.sqrt(2*self.area))*(2-self.poisson);
        #R = sqrt(LW) L = sqrt(km area)
        #print("StressChar",self.stressDrop_characteristic);
        if (not (pointT =="strike-dip" or pointT =="tensile" or pointT =="inflation")):
            raise Exception("Point Type: \""+pointT+"\" is invalid.");
        self.pointType = pointT;
        self.functionsDisplacement = None;
        self.functionsStressX = None;
        self.functionsStressY = None;
        self.functionsStressZ = None;
        self.rho = 2530;
        self.alpha = (self.lam + self.mu)/(self.lam+2*(self.mu));
        self.strikeAngle = strikeAngle; 
        self.normalvector = self.getNormalVector(self.strikeAngle,dipAngle);
        if dipAngle <0 or dipAngle > math.pi/2:
            raise Exception("Invalid Dip Angle. Must be from 0 to pi/2.");
        self.dip = dipAngle;
        self.rakeVector = self.getRakeVector(self.strikeAngle,self.dip, rake, self.normalvector);
        # if rake< -math.pi/2 or rake > math.pi/2:
        #     raise Exception("Invalid rake angle. Must be from -pi/2 to pi/2");
        if rake< -math.pi or rake > math.pi:
            raise Exception("Invalid rake angle. Must be from -pi to pi");
        self.rake = rake;
        self.UD = math.sin(rake);
        
        self.US = math.cos(rake);
        
        if(self.pointType == "strike-dip"):
            self.base_slip_rate = .01/(86400*365.25);
        else:
            self.base_slip_rate = 0;
        self.xi_state = 1;
        self.slip_rate = self.base_slip_rate;
        self.selfStiffness = 0;
        self.generateEquations();
        self.nodeIndex = nodeID;
        #self.distAdjust = 1000;
        self.distAdjust = 1;
        self.d = "";
        self.pressure = -1; #not used except for inflation nodes.


        self.shearCapSample = .3 *(self.rho * 9.81 * self.c); #setting a cap on self sampled shear stress as some fraction of isostatic pressure.

        
    def generateEquations(self):
        if (self.isDisplacement): #if you want displacement instead of stress tensor. Not really focusing on this rn
            if (self.pointType == "inflation"):
                self.__generateDisplacementInflation();
            #elif (self.pointType == "strike"):
            #    self.__generateDisplacementStrike();
            #elif (self.pointType == "dip"):
            #    self.__generateDisplacementDip();
            elif (self.pointType == "strike-dip"):
                self.__generateDisplacementStrikeDip();
            elif (self.pointType == "tensile"):
                self.__generateDisplacementTensile();
            else:
                raise Exception("Point Type: \""+self.pointType+"\" is invalid.");
        else:
            if (self.pointType == "inflation"):
                self.__generateStressXInflation();
                self.__generateStressYInflation();
                self.__generateStressZInflation();
                self.__generateDisplacementInflation();
            elif (self.pointType == "strike-dip"):
                self.__generateStressXStrikeDip();
                self.__generateStressYStrikeDip();
                self.__generateStressZStrikeDip();
                self.__generateDisplacementStrikeDip();
            # elif (self.pointType == "strike"):
            #     self.__generateStressXStrike();
            #     self.__generateStressYStrike();
            #     self.__generateStressZStrike();
            #     self.__generateDisplacementStrike();
            # elif (self.pointType == "dip"):
            #     self.__generateStressXDip();
            #     self.__generateStressYDip();
            #     self.__generateStressZDip();
            #     self.__generateDisplacementDip();
            elif (self.pointType == "tensile"):
                self.__generateStressXTensile();
                self.__generateStressYTensile();
                self.__generateStressZTensile();
                self.__generateDisplacementTensile();
                
            else:
                raise Exception("Point Type: \""+self.pointType+"\" is invalid.");
            

    #Constant Generation - Things based on evaluation constants
    #---------------------------------------------------------------------
    def set_d(self,z):
        self.d = self.c - z;
    def get_d(self):
        #return (self.c - z);
        #return self.c - z;
        return self.d
    def get_R2(self,x,y,z):
        return x**2 + y**2 + self.get_d()**2 + self.distAdjust**2;
    def get_Rx(self,x,y,z,power): #returns different powers of R
        return (self.get_R2(x,y,z))**(power/2.0);
    def get_p(self,y,z):
        return y*math.cos(self.dip) + self.get_d()*math.sin(self.dip);
    def get_q(self,y,z):
        return y*math.sin(self.dip) - self.get_d()*math.cos(self.dip);
    def get_s(self,p,q):
        return p*math.sin(self.dip) + q*math.cos(self.dip);
    def get_t(self,p,q):
        return p*math.cos(self.dip) - q*math.sin(self.dip);
    def get_A3(self,x,y,z):
        return 1 - 3*x**2/(self.get_R2(x, y, z));
    def get_A5(self,x,y,z):
        return 1 - 5*x**2/(self.get_R2(x, y, z));
    def get_A7(self,x,y,z):
        return 1 - 7*x**2/(self.get_R2(x, y, z));
    def get_B3(self,x,y,z):
        return 1 - 3*y**2/(self.get_R2(x, y, z));
    def get_B5(self,x,y,z):
        return 1-5*y**2/(self.get_R2(x, y, z));
    def get_B7(self,x,y,z):
        return 1-7*y**2/(self.get_R2(x, y, z));
    def get_C3(self,x,y,z):
        return 1-3*self.get_d()**2/(self.get_R2(x, y, z));
    def get_C5(self,x,y,z):
        return 1-5*self.get_d()**2/(self.get_R2(x, y, z));
    def get_C7(self,x,y,z):
        return 1-7*self.get_d()**2/(self.get_R2(x, y, z));
    
    def get_J1(self,x,y,z):
        R = self.get_Rx(x,y,z,1);
        R2 = self.get_R2(x,y,z);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,3);
        d = self.get_d();
        #if(R+d == 0):
            #print(R);
            #print(d);
        return -3*x*y*(  (3*R+d)/(R3*(R+d)**3)  - x**2 * (5*R2 + 4*R*d + d**2)/(R5 * (R+d)**4)  );
    
    def get_J2(self,x,y,z):
        R = self.get_Rx(x,y,z,1);
        R2 = self.get_R2(x,y,z);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,3);
        d = self.get_d();
        return (1/R3) - (3/(R*(R+d)**2)) + 3*(x**2)*(y**2)* (5*R2 +4*R*d+d**2)/(R5*(R+d)**4);
    
    def get_J3(self,x,y,z,J2):
        A3 = self.get_A3(x, y, z);
        R3 = self.get_Rx(x, y, z, 3);
        return A3/R3 - J2;
    
    def get_J4(self,x,y,z,J1):
        R5 = self.get_Rx(x,y,z,3);
        return -(3*x*y)/R5 - J1;
    
    def get_K1(self,x,y,z):
        R = self.get_Rx(x,y,z,1);
        R2 = self.get_R2(x,y,z);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,3);
        d = self.get_d();
        
        return -y* ( (2*R+d)/(R3*(R+d)**2) - x**2 *(8*R2 + 9*R*d + 3*d**2)/(R5*(R+d)**3));
    
    def get_K2(self,x,y,z):
        R = self.get_Rx(x,y,z,1);
        R2 = self.get_R2(x,y,z);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,3);
        d = self.get_d();
        
        return -x * ((2*R+d)/(R3*(R+d)**2) - y**2 * (8*R2 + 9*R*d +3*d**2)/(R5*(R+d)**3 ));
    
    def get_K3(self,x,y,z,K2):
        R5 = self.get_Rx(x,y,z,3);
        d = self.get_d();
        return -(3*x*d)/R5 - K2;
    
    def get_U(self,x,y,z,q):
        R2 = self.get_R2(x, y, z);
        return math.sin(self.dip) - 5*y*q/R2;
    
    def get_V(self,x,y,z,p,q,s):
        R2 = self.get_R2(x,y,z);
        return s - (5*y*p*q/R2);
    
    def get_W(self,U):
        return math.sin(self.dip) + U;
    
    def get_Uprime(self,x,y,z):
        R2 = self.get_R2(x, y, z);
        d = self.get_d();
        q = self.get_q(y, z);
        return math.cos(self.dip)+ (5*d*q/R2);
    
    def get_Vprime(self,x,y,z,p,q,t):
        R2 = self.get_R2(x, y, z);
        d = self.get_d();
        return t + (5*d*p*q/R2);
    
    def get_Wprime(self,Uprime):
        return math.cos(self.dip) + Uprime;
    
    
    
    def get_I1(self,x,y,z):
        R = self.get_Rx(x,y,z,1);
        R3 = self.get_Rx(x,y,z,3);
        d = self.get_d();
        return y*( 1/(R*(R+d)**2) - x**2*(3*R+d)/(R3*(R+d)**3));
    def get_I2(self,x,y,z):
        R = self.get_Rx(x,y,z,1);
        R3 = self.get_Rx(x,y,z,3);
        d = self.get_d();
        return x*( 1/(R*(R+d)**2) - y**2*(3*R+d)/(R3*(R+d)**3));
    def get_I3(self,x,y,z,I2):
        R3 = self.get_Rx(x,y,z,3);
        return (x/R3) -I2;
    def get_I4(self,x,y,z):
        R = self.get_Rx(x,y,z,1);
        R3 = self.get_Rx(x,y,z,3);
        d = self.get_d();
        return -x*y*(2*R+d)/(R3*(R+d)**2);
    def get_I5(self,x,y,z):
        R = self.get_Rx(x,y,z,1);
        R3 = self.get_Rx(x,y,z,3);
        d = self.get_d();
        return (1/(R*(R+d))) - x**2 *(2*R+d)/(R3*(R+d)**2);
    


    #Displacement
    #---------------------------------------------------------------------
    
    #do not use the map coordinate base for x,y,z. This is the frame of reference with the point at center. Requires generateEquations to be run first. Functions is a data structure holding the correct type of function for the source type. It's like putting together legos.
    def __evaluateDisplacement(self,x,y,z):
        if (self.functionsDisplacement ==None):
            raise Exception("ValueError: Functions are not instantiated. See generateEquations.");
        if self.pointType == "strike-dip":
            return self.US* ((self.M0)/(2*math.pi*self.mu) * (self.functionsDisplacement[0](x,y,z) - self.functionsDisplacement[0](x,y,-z) + self.functionsDisplacement[1](x,y,z) + z*self.functionsDisplacement[2](x,y,z))) + self.UD *((self.M0)/(2*math.pi*self.mu) * (self.functionsDisplacement[3](x,y,z) - self.functionsDisplacement[3](x,y,-z) + self.functionsDisplacement[4](x,y,z) + z*self.functionsDisplacement[5](x,y,z)));
        else:
            return (self.M0)/(2*math.pi*self.mu) * (self.functionsDisplacement[0](x,y,z) - self.functionsDisplacement[0](x,y,-z) + self.functionsDisplacement[1](x,y,z) + z*self.functionsDisplacement[2](x,y,z));
        
    
    #Strike/DIP
    
    def __generateDisplacementStrikeDip(self):
        self.functionsDisplacement = [self.__DispStrikeUA,self.__DispStrikeUB,self.__DispStrikeUC, self.__DispDipUA,self.__DispDipUB,self.__DispDipUC];
    
    #Strike
    def __generateDisplacementStrike(self):
        self.functionsDisplacement = [self.__DispStrikeUA,self.__DispStrikeUB,self.__DispStrikeUC];
        
    def __DispStrikeUA(self,x,y,z):
        #returns a 3 vector, X Y and Z components.
        q = self.get_q(y,z);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        Fx = (1-self.alpha)/2 * (q/R3) + (self.alpha/2)*(3*x**2*q/R5);
        Fy = (1-self.alpha)/2 * (x/R3) * math.sin(self.dip) + (self.alpha/2)*(3*x*y*q/R5);
        Fz = -(1-self.alpha)/2 * (x/R3)*math.cos(self.dip) + (self.alpha/2)*(3*x*self.get_d()*q/R5);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __DispStrikeUB(self,x,y,z):
        q = self.get_q(y,z);
        I1 = self.get_I1(x, y, z);
        I2 = self.get_I2(x, y, z);
        I4 = self.get_I4(x, y, z);
        R5 = self.get_Rx(x,y,z,5);
        
        Fx = (-3*x**2*q/R5) - (1-self.alpha)/self.alpha *I1*math.sin(self.dip);
        Fy = (-3*x*y*q/R5) - (1-self.alpha)/self.alpha *I2*math.sin(self.dip);
        Fz = (-3*self.c*x*q/R5) - (1-self.alpha)/self.alpha *I4*math.sin(self.dip);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __DispStrikeUC(self,x,y,z):
        q = self.get_q(y,z);
        R2 = self.get_Rx(x,y,z,2);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        A3 = self.get_A3(x, y, z);
        A5 = self.get_A5(x, y, z);
        
        Fx = -(1-self.alpha)*A3/R3*math.cos(self.dip) + self.alpha * (3*self.c*q/R5) * A5;
        Fy = (1-self.alpha) *(3*x*y/R5)*math.cos(self.dip) + self.alpha *(3*self.c*x/R5) * (math.sin(self.dip) - (5*y*q/R2));
        Fz = -(1-self.alpha)*(3*x*y/R5)*math.sin(self.dip) + self.alpha * (3*self.c*x/R5) * (math.cos(self.dip) + 5*self.get_d()*q/R2);
        
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    #dip
    
    def __generateDisplacementDip(self):
        self.functionsDisplacement = [self.__DispDipUA,self.__DispDipUB,self.__DispDipUC];
        
    def __DispDipUA(self,x,y,z):
        q = self.get_q(y,z);
        p = self.get_p(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        Fx = (self.alpha/2) * (3*x*p*q/R5);
        Fy = (1-self.alpha)/2 * s/R3 + (self.alpha/2) * (3*y*p*q/R5);
        Fz = -(1-self.alpha)/2 * t/R3 + (self.alpha/2) * (3*self.get_d()*p*q/R5);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __DispDipUB(self,x,y,z):
        q = self.get_q(y,z);
        p = self.get_p(y, z);
        R5 = self.get_Rx(x,y,z,5);
        I1 = self.get_I1(x, y, z);
        I2 = self.get_I2(x, y, z);
        I3 = self.get_I3(x, y, z, I2);
        I5 = self.get_I5(x, y, z);
        Fx = -(3*x*p*q/R5) + (1-self.alpha)/self.alpha * I3*math.sin(self.dip)*math.cos(self.dip);
        Fy = -(3*y*p*q/R5) + (1-self.alpha)/self.alpha * I1*math.sin(self.dip)*math.cos(self.dip);
        Fz = -(3*self.c*p*q/R5) + (1-self.alpha)/self.alpha * I5*math.sin(self.dip)*math.cos(self.dip);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __DispDipUC(self,x,y,z):
        q = self.get_q(y,z);
        p = self.get_p(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);
        R2 = self.get_Rx(x,y,z,2);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x,y,z,7);
        
        Fx = (1-self.alpha)*(3*x*t/R5) - self.alpha * (15*self.c*x*p*q/R7);
        Fy = -(1-self.alpha)* (1/R3) *(math.cos(2*self.dip) - (3*y*t/R2)) + self.alpha * (3*self.c/R5)*(s-(5*y*p*q/R2));
        Fz = -(1-self.alpha)*(self.get_A3(x, y, z)/R3)*math.sin(self.dip)*math.cos(self.dip) + self.alpha * (3*self.c/R5) * (t+ (5*self.get_d()*p*q/R2));
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    
    #Tensile
    
    def __generateDisplacementTensile(self):
        self.functionsDisplacement = [self.__DispTensileUA,self.__DispTensileUB,self.__DispTensileUC];
    
    def __DispTensileUA(self,x,y,z):
        q = self.get_q(y,z);
        p = self.get_p(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);

        Fx = (1-self.alpha)/2 * (x/R3) - self.alpha/2 * (3*x*q**2)/R5;
        Fy = (1-self.alpha)/2 * (t/R3) - self.alpha/2 * (3*y*q**2)/R5;
        Fz = (1-self.alpha)/2 * (s/R3) - self.alpha/2 * (3*self.get_d()*q**2)/R5;
        
        #return np.array([Fx, 2*self.M0 * Fy, self.lam/self.mu * self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __DispTensileUB(self,x,y,z):
        q = self.get_q(y,z);
        R5 = self.get_Rx(x,y,z,5);
        I1 = self.get_I1(x, y, z);
        I2 = self.get_I2(x, y, z)
        I3 = self.get_I3(x, y, z, I2);
        I5 = self.get_I5(x, y, z);
        
        Fx = 3*x*q**2/R5 - (1-self.alpha)/self.alpha * I3 * math.sin(self.dip)**2;
        Fy = 3*y*q**2/R5 - (1-self.alpha)/self.alpha * I1 * math.sin(self.dip)**2;
        Fz = 3*self.c*q**2/R5 - (1-self.alpha)/self.alpha * I5 *math.sin(self.dip)**2;
        
        #return np.array([Fx, 2*self.M0 * Fy, self.lam/self.mu * self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __DispTensileUC(self,x,y,z):
        q = self.get_q(y,z);
        p = self.get_p(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);
        R2 = self.get_Rx(x,y,z,2);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x,y,z,7);
        A3 = self.get_A3(x, y, z)
        
        Fx = -(1-self.alpha ) * 3*x*s/R5 + self.alpha * (15*self.c * x * q**2)/R7 - self.alpha *(3*x*z/R5);
        Fy = (1-self.alpha) * (1/R3) * (math.sin(2* self.dip) - (3*y*s/R2)) + self.alpha *(3*self.c/R5)* (t - y + (5*y*q**2/R2)) - self.alpha *(3*y*z/R5);
        Fz = -(1-self.alpha)*(1/R3) * (1 - A3*math.sin(self.dip)**2) - self.alpha * (3*self.c/R5) * (s-self.get_d()+(5*self.get_d()*q**2/R2)) + self.alpha * (3*self.get_d()*z)/R5;
        #return np.array([Fx, 2*self.M0 * Fy, self.lam/self.mu * self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
        
    
    #Inflation
    def __generateDisplacementInflation(self):
        self.functionsDisplacement = [self.__DispInflationUA,self.__DispInflationUB,self.__DispInflationUC];
    
    def __DispInflationUA(self,x,y,z):
        R3 = self.get_Rx(x,y,z,3);
        Fx = -(1-self.alpha)/2 * x/R3;
        Fy = -(1-self.alpha)/2 * y/R3;
        Fz = -(1-self.alpha)/2 * self.get_d()/R3;
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __DispInflationUB(self,x,y,z):
        R3 = self.get_Rx(x,y,z,3);
        Fx = (1-self.alpha)/self.alpha * x/R3;
        Fy = (1-self.alpha)/self.alpha * y/R3;
        Fz = (1-self.alpha)/self.alpha * self.get_d()/R3;
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __DispInflationUC(self,x,y,z):
        d = self.get_d();
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        C3 = self.get_C3(x, y, z);
        Fx = (1-self.alpha)*(3*x*d/R5);
        Fy = (1-self.alpha)*(3*y*d/R5);
        Fz = (1-self.alpha)*(C3/R3);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    
    
    
    #X Stress
    #------------------------------------------------------------------------------------------
    
    def __evaluateStressX(self,x,y,z):
        if (self.functionsStressX ==None):
            raise Exception("ValueError: Functions are not instantiated. See generateEquations.");
        if (self.pointType == "strike-dip"):
            return self.US*((self.M0)/(2*math.pi*self.mu) * (self.functionsStressX[0](x,y,z) - self.functionsStressX[0](x,y,-z) + self.functionsStressX[1](x,y,z) + z*self.functionsStressX[2](x,y,z))) + self.UD *((self.M0)/(2*math.pi*self.mu) * (self.functionsDisplacement[3](x,y,z) - self.functionsDisplacement[3](x,y,-z) + self.functionsDisplacement[4](x,y,z) + z*self.functionsDisplacement[5](x,y,z)));
        else:
            return (self.M0)/(2*math.pi*self.mu) * (self.functionsStressX[0](x,y,z) - self.functionsStressX[0](x,y,-z) + self.functionsStressX[1](x,y,z) + z*self.functionsStressX[2](x,y,z));
    
    
    #StrikeDip
    
    def __generateStressXStrikeDip(self):
        self.functionsStressX = [self.__StressXStrikeUA,self.__StressXStrikeUB,self.__StressXStrikeUC,self.__StressXDipUA,self.__StressXDipUB,self.__StressXDipUC];
    
    #Inflation
    def __generateStressXInflation(self):
        self.functionsStressX = [self.__StressXInflationUA,self.__StressXInflationUB,self.__StressXInflationUC];
    
    def __StressXInflationUA(self,x,y,z):
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        A3 = self.get_A3(x, y, z);
        Fx = -(1-self.alpha)/2 * A3/R3;
        Fy = (1-self.alpha)/2 * (3*x*y)/R5;
        Fz = (1-self.alpha)/2 * (3*x*self.get_d()/R5);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressXInflationUB(self,x,y,z):
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        A3 = self.get_A3(x, y, z);
        
        Fx = (1-self.alpha)/self.alpha *A3/R3;
        Fy = -(1-self.alpha)/self.alpha *(3*x*y/R5);
        Fz = -(1-self.alpha)/self.alpha *(3*x*self.get_d()/R5);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressXInflationUC(self,x,y,z):
        R7 = self.get_Rx(x,y,z,7);
        R5 = self.get_Rx(x,y,z,5);
        A5 = self.get_A5(x, y, z);
        C5 = self.get_C5(x, y, z);
        d = self.get_d();
        
        Fx = (1-self.alpha)* (3*d/R5)*A5;
        Fy = -(1-self.alpha)*(15*x*y*d/R7);
        Fz = -(1-self.alpha)*(3*x/R5)*C5;
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);

    
    #Strike
    def __generateStressXStrike(self):
        self.functionsStressX = [self.__StressXStrikeUA,self.__StressXStrikeUB,self.__StressXStrikeUC];
        
    def __StressXStrikeUA(self,x,y,z):
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        A3 = self.get_A3(x, y, z);
        A5 = self.get_A5(x,y,z);
        q = self.get_q(y, z);
        
        Fx = -(1-self.alpha)/2 * (3*x*q)/R5 + (self.alpha/2)*(3*x*q/R5)*(1+A5);
        Fy = (1-self.alpha)/2 * A3/R3 * math.sin(self.dip) + (self.alpha/2) * (3*y*q/R5) * A5;
        Fz = -(1-self.alpha)/2 * A3/R3*math.cos(self.dip) + (self.alpha/2)*(3*self.get_d()*q/R5)*A5;
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
        
    def __StressXStrikeUB(self,x,y,z):
        R5 = self.get_Rx(x,y,z,5);
        A5 = self.get_A5(x, y, z);
        q = self.get_q(y, z);
        J1 = self.get_J1(x, y, z);
        J2 = self.get_J2(x, y, z);
        K1 = self.get_K1(x, y, z);
        
        Fx = -(3*x*q/R5)*(1+A5) - (1-self.alpha)/self.alpha * J1*math.sin(self.dip);
        Fy = -(3*y*q/R5)*A5 - (1-self.alpha)/self.alpha *J2 *math.sin(self.dip);
        Fz = -(3*self.c*q)/R5 *A5 - (1-self.alpha)/self.alpha * K1*math.sin(self.dip);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressXStrikeUC(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x,y,z,7);
        A5 = self.get_A5(x, y, z);
        A7 = self.get_A7(x, y, z);
        q = self.get_q(y, z);
        d = self.get_d();
        
        Fx = (1-self.alpha)*(3*x/R5)*(2+A5)*math.cos(self.dip) - self.alpha * (15*self.c*x*q/R7) *(2+A7);
        Fy = (1-self.alpha)*(3*y/R5)*A5*math.cos(self.dip) + self.alpha*(3*self.c/R5)*(A5*math.sin(self.dip) - (5*y*q/R2)*A7);
        Fz = -(1-self.alpha)*(3*y/R5)*A5*math.sin(self.dip) + self.alpha*(3*self.c/R5)*(A5*math.cos(self.dip) + (5*d*q/R2)*A7);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    #DIP
    
    def __generateStressXDip(self):
        self.functionsStressX = [self.__StressXDipUA, self.__StressXDipUB,self.__StressXDipUC];
        
    def __StressXDipUA(self,x,y,z):
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x,y,z,7);
        A5 = self.get_A5(x, y, z);
        p = self.get_p(y, z);
        q = self.get_q(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);
        d = self.get_d();
        
        
        
        Fx = (self.alpha)/2 *  (3*p*q/R5)*A5;
        Fy = -(1-self.alpha)/2 * (3*x*s/R5) - (self.alpha/2) * (15*x*y*p*q/R7);
        Fz = (1-self.alpha)/2 * (3*x*t/R5) - (self.alpha/2) * (15*x*d*p*q/R7);
        
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressXDipUB(self,x,y,z):
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x,y,z,7);
        A5 = self.get_A5(x, y, z);
        p = self.get_p(y, z);
        q = self.get_q(y, z);
        
        J1 = self.get_J1(x, y, z);
        J2 = self.get_J2(x, y, z);
        J3 = self.get_J3(x, y, z, J2);
        K2 = self.get_K2(x, y, z);
        K3 = self.get_K3(x, y, z, K2);
        
        Fx = -(3*p*q/R5)*A5 + (1-self.alpha)/self.alpha * J3 *math.sin(self.dip)*math.cos(self.dip);
        Fy = (15*x*y*p*q)/R7+ (1-self.alpha)/self.alpha * J1 * math.sin(self.dip)*math.cos(self.dip);
        Fz = (15*self.c*x*p*q/R7) + (1-self.alpha)/self.alpha * K3 * math.sin(self.dip)*math.cos(self.dip);
    
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressXDipUC(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x,y,z,7);
        A5 = self.get_A5(x, y, z);
        A7 = self.get_A7(x, y, z);
        p = self.get_p(y, z);
        q = self.get_q(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);
        d = self.get_d();


        
        Fx = (1-self.alpha)*(3*t/R5)*A5 - (self.alpha)*(15*self.c*p*q/R7)*A7;
        Fy = (1-self.alpha)*(3*x/R5)*(math.cos(2*self.dip) - (5*y*t/R2)) - self.alpha * (15*self.c*x/R7) * (s-(7*y*p*q/R2));
        Fz = (1-self.alpha)*(3*x/R5)*(2+A5)*math.sin(self.dip)*math.cos(self.dip) - self.alpha * (15*self.c*x/R7)*(t+(7*d*p*q/R2));
        
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    
    #tensile
    
    def __generateStressXTensile(self):
        self.functionsStressX = [self.__StressXTensileUA, self.__StressXTensileUB,self.__StressXTensileUC];
    
    
    def __StressXTensileUA(self,x,y,z):
        q = self.get_q(y,z);
        p = self.get_p(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x,y,z,7);
        A3 = self.get_A3(x, y, z);
        A5 = self.get_A5(x, y, z);
        
        Fx = (1-self.alpha)/2 * A3/R3 * - self.alpha/2 * (3*q**2/R5)*A5;
        Fy = -(1-self.alpha)/2 * (3*x*t/R5) + (self.alpha/2) *(15*x*y*q**2)/R7;
        Fz = -(1-self.alpha)/2 * (3*x*s/R5) + (self.alpha/2) *(15*x*self.get_d()*q**2)/R7;
        #return np.array([Fx,2*self.M0*Fy,self.lam/self.mu *  self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressXTensileUB(self,x,y,z):
        q = self.get_q(y,z);
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x,y,z,7);
        A5 = self.get_A5(x, y, z);
        J1 = self.get_J1(x, y, z);
        J2 = self.get_J2(x, y, z);
        J3 = self.get_J3(x, y, z, J2);
        
        K2 = self.get_K2(x, y, z);
        K3 = self.get_K3(x, y, z, K2);
        
        Fx = (3*q**2/R5)*A5 - (1-self.alpha)/self.alpha * J3*math.sin(self.dip)**2;
        Fy = -(15*x*y*q**2)/R7 - (1-self.alpha)/self.alpha *J1*math.sin(self.dip)**2;
        Fz = -(15*self.c*x*q**2)/R7 - (1-self.alpha)/self.alpha *K3 * math.sin(self.dip)**2;
        #return np.array([Fx,2*self.M0*Fy,self.lam/self.mu * self.M0* Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressXTensileUC(self,x,y,z):
        q = self.get_q(y,z);
        p = self.get_p(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);
        R2 = self.get_Rx(x,y,z,2);
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x,y,z,7);
        A5 = self.get_A5(x, y, z);
        A7 = self.get_A7(x, y, z);
        
        
        Fx = -(1-self.alpha) * 3*s/R5 *A5 + self.alpha * (15*self.c*q**2)/R7 * A7 - self.alpha*(3*z/R5)*A5;
        Fy = -(1-self.alpha) * 3*x/R5*(math.sin(2*self.dip) - 5*y*s/R2) - self.alpha *(15*self.c*x/R7) * (t-y+(7*y*q**2/R2)) + self.alpha * (15*x*y*z/R7)
        Fz = (1-self.alpha) * (3*x/R5)*(1 - (2+A5)*math.sin(self.dip)**2) + self.alpha * (15*self.c*x/R7) * (s - self.get_d() + (7*self.get_d()*q**2)/R2) - self.alpha*(15*x*self.get_d()*z/R7);
        
        #return np.array([Fx,2*self.M0*Fy,self.lam/self.mu * self.M0* Fz]);
        return np.array([Fx,Fy,Fz]);
    
    
    #Y Stress
    #--------------------------------------------------------------------------------------------
    def __evaluateStressY(self,x,y,z):
        if (self.functionsStressY ==None):
            raise Exception("ValueError: Functions are not instantiated. See generateEquations.");
        if (self.pointType == "strike-dip"):
            return self.US*((self.M0)/(2*math.pi*self.mu) * (self.functionsStressY[0](x,y,z) - self.functionsStressY[0](x,y,-z) + self.functionsStressY[1](x,y,z) + z*self.functionsStressY[2](x,y,z)))+ self.UD*((self.M0)/(2*math.pi*self.mu) * (self.functionsStressY[3](x,y,z) - self.functionsStressY[3](x,y,-z) + self.functionsStressY[4](x,y,z) + z*self.functionsStressY[5](x,y,z)));
        else:  
            return (self.M0)/(2*math.pi*self.mu) * (self.functionsStressY[0](x,y,z) - self.functionsStressY[0](x,y,-z) + self.functionsStressY[1](x,y,z) + z*self.functionsStressY[2](x,y,z));
    
    #StrikeDIP
    def __generateStressYStrikeDip(self):
        self.functionsStressY = [self.__StressYStrikeUA,self.__StressYStrikeUB,self.__StressYStrikeUC,self.__StressYDipUA,self.__StressYDipUB,self.__StressYDipUC];
    
    #Inflation
    def __generateStressYInflation(self):
        self.functionsStressY = [self.__StressYInflationUA,self.__StressYInflationUB,self.__StressYInflationUC];
        
    def __StressYInflationUA(self,x,y,z):
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        B3 = self.get_B3(x, y, z);
        d = self.get_d();
        
        Fx = (1-self.alpha)/2 * (3*x*y/R5);
        Fy = -(1-self.alpha)/2 * (B3/R3);
        Fz = (1-self.alpha)/2 * (3*y*d/R5);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressYInflationUB(self,x,y,z):
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        B3 = self.get_B3(x, y, z);
        d = self.get_d();
        
        Fx = -(1-self.alpha)/self.alpha * (3*x*y/R5);
        Fy = (1-self.alpha)/self.alpha * (B3/R3);
        Fz = -(1-self.alpha)/self.alpha * (3*y*d/R5);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressYInflationUC(self,x,y,z):
        R7 = self.get_Rx(x,y,z,7);
        R5 = self.get_Rx(x,y,z,5);
        B5 = self.get_B5(x, y, z);
        C5 = self.get_C5(x, y, z);
        d = self.get_d();
        
        Fx = -(1-self.alpha)*(15*x*y*d/R7);
        Fy = (1-self.alpha) * (3*d/R5)*B5;
        Fz = -(1-self.alpha) * (3*y/R5)*C5;
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    #Strike
    def __generateStressYStrike(self):
        self.functionsStressY = [self.__StressYStrikeUA,self.__StressYStrikeUB,self.__StressYStrikeUC];
        
    def __StressYStrikeUA(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        q = self.get_q(y, z);
        U = self.get_U(x, y, z, q);
        d = self.get_d();
        
        Fx = (1-self.alpha)/2 * (1/R3) * (math.sin(self.dip) - (3*y*q/R2)) + (self.alpha/2) * (3*x**2/R5)*U;
        Fy = -(1-self.alpha)/2 *(3*x*y/R5) * math.sin(self.dip) + (self.alpha/2) * (3*x*y/R5) * U + (self.alpha)/2 * (3*x*q/R5);
        Fz = (1-self.alpha)/2 * (3*x*y/R5) * math.cos(self.dip) + (self.alpha/2) *(3*x*d/R5) * U;
        
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressYStrikeUB(self,x,y,z):
        R5 = self.get_Rx(x,y,z,5);
        q = self.get_q(y, z);
        U = self.get_U(x, y, z, q);
        
        J1 = self.get_J1(x, y, z);
        J2 = self.get_J2(x, y, z);
        J4 = self.get_J4(x, y, z, J1);
        K2 = self.get_K2(x, y, z);
        
        Fx = -(3*x**2/R5)*U - (1-self.alpha)/self.alpha *J2*math.sin(self.dip);
        Fy = -(3*x*y/R5)*U - (3*x*q/R5) - (1-self.alpha)/self.alpha * J4 *math.sin(self.dip);
        Fz = -(3*self.c*x/R5)*U - (1-self.alpha)/self.alpha *K2*math.sin(self.dip);
        
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressYStrikeUC(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x, y, z, 7);
        A5 = self.get_A5(x, y, z);
        A7 = self.get_A7(x, y, z)
        q = self.get_q(y, z);
        d = self.get_d();
        B5 = self.get_B5(x, y, z);
        B7 = self.get_B7(x, y, z);
        C7 = self.get_C7(x, y, z);
        
        Fx = (1-self.alpha) * (3*y/R5) * A5 * math.cos(self.dip) + self.alpha *(3*self.c/R5)* (A5*math.sin(self.dip) - (5*y*q/R2)*A7);
        Fy = (1-self.alpha)*(3*x/R5) *B5 *math.cos(self.dip) - self.alpha * (15*self.c*x/R7)*(2*y*math.sin(self.dip) + q*B7);
        Fz = -(1-self.alpha)*(3*x/R5) *B5*math.sin(self.dip) + self.alpha * (15*self.c*x/R7)*(d*B7*math.sin(self.dip) - y*C7*math.cos(self.dip));
        
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __generateStressYDip(self):
        self.functionsStressY = [self.__StressYDipUA,self.__StressYDipUB,self.__StressYDipUC];
        
    
    def __StressYDipUA(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        p = self.get_p(y, z);
        q = self.get_q(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);
        d = self.get_d();
        
        V = self.get_V(x, y, z, p, q, s);

        
        Fx = (self.alpha/2)*(3*x/R5)*V;
        Fy = (1-self.alpha)/2 * (1/R3) * (math.sin(2*self.dip) - (3*y*s/R2)) + (self.alpha)/2 *(3*y/R5)*V + (self.alpha/2)*(3*p*q/R5);
        Fz = -(1-self.alpha)/2 * (1/R3) * (math.cos(2*self.dip) - (3*y*t/R2)) + (self.alpha)/2 * (3*d/R5) * V;
        
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressYDipUB(self,x,y,z):
        R5 = self.get_Rx(x,y,z,5);
        p = self.get_p(y, z);
        q = self.get_q(y, z);
        s = self.get_s(p, q);
        
        V = self.get_V(x, y, z, p, q, s);
        
        J1 = self.get_J1(x, y, z);
        J2 = self.get_J2(x, y, z);

        K1 = self.get_K1(x, y, z);
        
        Fx = -(3*x/R5)*V + (1-self.alpha)/self.alpha * J1 *math.sin(self.dip)*math.cos(self.dip);
        Fy = -(3*y/R5)*V + (-3*p*q/R5) + (1-self.alpha)/self.alpha * J2*math.sin(self.dip)*math.cos(self.dip);
        Fz = -(3*self.c/R5)*V + (1-self.alpha)/self.alpha *K1*math.sin(self.dip)*math.cos(self.dip);
        
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressYDipUC(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R4 = self.get_Rx(x,y,z,4);
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x,y,z,7);
        A5 = self.get_A5(x, y, z);
        p = self.get_p(y, z);
        q = self.get_q(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);
        d = self.get_d();
        
        
        B5 = self.get_B5(x, y, z);
        B7 = self.get_B7(x, y, z);

        
        Fx = (1-self.alpha)*(3*x/R5)*(math.cos(2*self.dip) - (5*y*t/R2)) - self.alpha *(15*self.c*x/R7)*(s - (7*y*p*q/R2));
        Fy = (1-self.alpha)*(3/R5)*(2*y*math.cos(2*self.dip) + t*B5) + self.alpha *(3*self.c/R5) * (math.sin(2*self.dip) - (10*y*s/R2) - (5*p*q/R2)*B7);
        Fz = (1-self.alpha)*(3*y/R5)*A5*math.sin(self.dip)*math.cos(self.dip) - self.alpha * (3*self.c/R5)*( (3+A5)*math.cos(2*self.dip) + (35*y*d*p*q/R4));
        
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    #Tensile
    
    def __generateStressYTensile(self):
        self.functionsStressY = [self.__StressYTensileUA,self.__StressYTensileUB,self.__StressYTensileUC];
        
    def __StressYTensileUA(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        p = self.get_p(y, z);
        q = self.get_q(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);

        U = self.get_U(x, y, z, q);
        W = self.get_W(U);
        
        Fx = -(1-self.alpha)/2 * (3*x*y/R5) - self.alpha/2 * (3*x*q/R5)*W;
        Fy = (1-self.alpha)/2 * (1/R3) * (math.cos(2*self.dip) -3*y*t/R2) - self.alpha/2 * (3*y*q/R5) * W - (self.alpha/2) * 3*q**2/R5;
        Fz = (1-self.alpha)/2 * (1/R3) * (math.sin(2*self.dip) -3*y*s/R2) - self.alpha/2 * (3*self.get_d()*q/R5)*W;
        #return np.array([Fx,2*self.M0*Fy,self.lam/self.mu * self.M0* Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressYTensileUB(self,x,y,z):
        R5 = self.get_Rx(x,y,z,5);
        q = self.get_q(y, z);
        
        
        
        J1 = self.get_J1(x, y, z);
        J2 = self.get_J2(x, y, z);
        K1 = self.get_K1(x, y, z);
        U = self.get_U(x, y, z, q);
        W = self.get_W(U);
        
        Fx = (3*x*q/R5)*W - (1-self.alpha)/self.alpha *J1*math.sin(self.dip)**2;
        Fy = (3*y*q/R5*W) + 3*q**2/R5 - (1-self.alpha)/self.alpha *J2 *math.sin(self.dip)**2;
        Fz = (3*self.c*q/R5)*W - (1-self.alpha)/self.alpha *K1*math.sin(self.dip)**2;
        #return np.array([Fx,2*self.M0*Fy,self.lam/self.mu * self.M0* Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressYTensileUC(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x,y,z,7);
        A5 = self.get_A5(x, y, z);
        p = self.get_p(y, z);
        q = self.get_q(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);
        
        
        B5 = self.get_B5(x, y, z);
        B7 = self.get_B7(x, y, z);
        
        
        Fx = -(1-self.alpha)*(3*x/R5)*(math.sin(2*self.dip) - 5*y*s/R2) - self.alpha*(15*self.c*x/R7)*(t-y+7*y*q**2/R2) + self.alpha * 15*x*y*z/R7;
        Fy = -(1-self.alpha)*3/R5*(2*y*math.sin(2*self.dip) +s*B5) - self.alpha * (3*self.c/R5)*(2*math.sin(self.dip)**2 + (10*y*(t-y)/R2) - 5*q**2/R2*B7) - self.alpha * (3*z/R5*B5);
        Fz = (1-self.alpha)*(3*y/R5)*(1-A5*math.sin(self.dip)**2)+self.alpha*(3*self.c/R5)*( (3+A5)*math.sin(2*self.dip) - 5*y*self.get_d()/R2*(2-7*q**2/R2)) - self.alpha*(15*y*self.get_d()*z/R7);
        #return np.array([Fx,2*self.M0*Fy,self.lam/self.mu * self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    #Z Stress
    #-----------------------------------------------------------------------------------------
    
    def __evaluateStressZ(self,x,y,z):
        if (self.functionsStressZ ==None):
            raise Exception("ValueError: Functions are not instantiated. See generateEquations.");
        if (self.functionsDisplacement == None):
            raise Exception("ValueError: Functions are not instantiated. See generateEquations.");
        
        if (self.pointType == "strike-dip"):
            return self.US * ((self.M0)/(2*math.pi*self.mu) * (self.functionsStressZ[0](x,y,z) - self.functionsStressZ[0](x,y,-z) + self.functionsStressZ[1](x,y,z) +self.functionsDisplacement[2](x,y,z) +  z*self.functionsStressZ[2](x,y,z))) + self.UD *(self.M0)/(2*math.pi*self.mu) * (self.functionsStressZ[3](x,y,z) - self.functionsStressZ[3](x,y,-z) + self.functionsStressZ[4](x,y,z) +self.functionsDisplacement[5](x,y,z) +  z*self.functionsStressZ[5](x,y,z));
        
        else:
            return (self.M0)/(2*math.pi*self.mu) * (self.functionsStressZ[0](x,y,z) - self.functionsStressZ[0](x,y,-z) + self.functionsStressZ[1](x,y,z) +self.functionsDisplacement[2](x,y,z) +  z*self.functionsStressZ[2](x,y,z));
    
    #StrikeDIP
    
    def __generateStressZStrikeDip(self):
        self.functionsStressZ = [self.__StressZStrikeUA,self.__StressZStrikeUB,self.__StressZStrikeUC,self.__StressZDipUA,self.__StressZDipUB,self.__StressZDipUC];
    
    
    #inflation
    def __generateStressZInflation(self):
        self.functionsStressZ = [self.__StressZInflationUA,self.__StressZInflationUB,self.__StressZInflationUC];
        
    def __StressZInflationUA(self,x,y,z):
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        C3 = self.get_C3(x, y, z);
        d = self.get_d();
        
        Fx = -(1-self.alpha)/2 * (3*x*d/R5);
        Fy = -(1-self.alpha)/2 * (3*y*d/R5);
        Fz = (1-self.alpha)/2 * C3/R3;
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressZInflationUB(self,x,y,z):
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        C3 = self.get_C3(x, y, z);
        d = self.get_d();
        
        Fx = (1-self.alpha)/self.alpha *(3*x*d/R5);
        Fy = (1-self.alpha)/self.alpha *(3*y*d/R5);
        Fz = -(1-self.alpha)/self.alpha *(C3/R3);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressZInflationUC(self,x,y,z):
        R5 = self.get_Rx(x,y,z,5);
        C5 = self.get_C5(x, y, z);
        d = self.get_d();
        
        Fx = -(1-self.alpha)*(3*x/R5)*C5;
        Fy = -(1-self.alpha)*(3*y/R5)*C5;
        Fz = (1-self.alpha)*(3*d/R5)*(2+C5);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    
    #strike
    def __generateStressZStrike(self):
        self.functionsStressZ = [self.__StressZStrikeUA,self.__StressZStrikeUB,self.__StressZStrikeUC];
    
    def __StressZStrikeUA(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        q = self.get_q(y, z);
        d = self.get_d();
        
        Uprime = self.get_Uprime(x, y, z);
        
        Fx = (1-self.alpha)/2 * (1/R3) * (math.cos(self.dip) + (3*d*q/R2)) + (self.alpha/2)*(3*x**2/R5)*Uprime;
        Fy = (1-self.alpha)/2 *(3*x*d/R5)*math.sin(self.dip) + (self.alpha/2)*(3*x*y/R5)*Uprime;
        Fz = -(1-self.alpha)/2 *(3*x*d/R5)*math.cos(self.dip) + (self.alpha/2)*(3*x*d/R5)*Uprime - (self.alpha/2) *(3*x*q/R5);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    
    def __StressZStrikeUB(self,x,y,z):
        R5 = self.get_Rx(x,y,z,5);
        Uprime = self.get_Uprime(x, y, z);
        K1 = self.get_K1(x, y, z);
        K2 = self.get_K2(x, y, z);
        
        Fx = -(3*x**2/R5)*Uprime + (1-self.alpha)/self.alpha *K1*math.sin(self.dip);
        Fy = -(3*x*y/R5)*Uprime + (1-self.alpha)/self.alpha *K2*math.sin(self.dip);
        Fz = -(3*self.c*x/R5)*Uprime + (1-self.alpha)/self.alpha* (3*x*y/R5)*math.sin(self.dip);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressZStrikeUC(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x, y, z, 7);
        A5 = self.get_A5(x, y, z);
        A7 = self.get_A7(x, y, z)
        q = self.get_q(y, z);
        d = self.get_d();
        B7 = self.get_B7(x, y, z);
        C7 = self.get_C7(x, y, z);
        
        
        Fx = -(1-self.alpha)*(3*d/R5)*A5*math.cos(self.dip) + self.alpha*(3*self.c/R5)*(A5*math.cos(self.dip) + (5*d*q/R2) *A7);
        Fy = (1-self.alpha)*(15*x*y*d/R7)*math.cos(self.dip) + self.alpha*(15*self.c*x/R7)*(d*B7*math.sin(self.dip) - y*C7*math.cos(self.dip) );
        Fz = -(1-self.alpha)*(15*x*y*d/R7)*math.sin(self.dip) + self.alpha*(15*self.c*x/R7)*(2*d*math.cos(self.dip) - q *C7);
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    
    def __generateStressZDip(self):
        self.functionsStressZ = [self.__StressZDipUA, self.__StressZDipUB, self.__StressZDipUC];
        
    def __StressZDipUA(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        p = self.get_p(y, z);
        q = self.get_q(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);
        d = self.get_d();
        
        Vprime = self.get_Vprime(x, y, z, p, q, t);
        
        
        Fx = (self.alpha)/2 *(3*x/R5)*Vprime;
        Fy = (1-self.alpha)/2 *(1/R3)* (math.cos(2*self.dip) + (3*d*s/R2))  + (self.alpha/2)*(3*y/R5)*Vprime;
        Fz = (1-self.alpha)/2 *(1/R3)*(math.sin(2*self.dip) - (3*d*t/R2)) + (self.alpha/2)*(3*d/R5)*Vprime - (self.alpha/2)*(3*p*q/R5);
        
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressZDipUB(self,x,y,z):
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        A3 = self.get_A3(x, y, z);
        p = self.get_p(y, z);
        q = self.get_q(y, z);
        t = self.get_t(p, q);
        
        Vprime = self.get_Vprime(x, y, z, p, q, t);
        
        K1 = self.get_K1(x, y, z);
        K2 = self.get_K2(x, y, z);
        K3 = self.get_K3(x, y, z, K2);
        
        Fx = -(3*x/R5)*Vprime - (1-self.alpha)/self.alpha *K3*math.sin(self.dip)*math.cos(self.dip);
        Fy = -(3*y/R5)*Vprime - (1-self.alpha)/self.alpha *K1*math.sin(self.dip)*math.cos(self.dip);
        Fz = -(3*self.c/R5)*Vprime + (1-self.alpha)/self.alpha *A3/R3*math.sin(self.dip)*math.cos(self.dip);
        
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressZDipUC(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R4 = self.get_Rx(x,y,z,4);
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x,y,z,7);
        A5 = self.get_A5(x, y, z);
        p = self.get_p(y, z);
        q = self.get_q(y, z);
        t = self.get_t(p, q);
        d = self.get_d();
        
        
        B5 = self.get_B5(x, y, z);
        C5 = self.get_C5(x, y, z);
        C7 = self.get_C7(x, y, z);


        
        Fx = -(1-self.alpha)*(3*x/R5)*(math.sin(2*self.dip) - (5*d*t/R2)) - (self.alpha)*(15*self.c*x/R7)*(t+(7*d*p*q/R2));
        Fy = -(1-self.alpha)*(3/R5)*(d*B5*math.cos(2*self.dip) +y*C5*math.sin(2*self.dip)) - self.alpha *(3*self.c/R5)*((3+A5)*math.cos(2*self.dip) +(35*y*d*p*q/R4));
        Fz = -(1-self.alpha)*(3*d/R5)*A5*math.sin(self.dip)*math.cos(self.dip) - (self.alpha)*(3*self.c/R5)*(math.sin(2*self.dip) - (10*d*t/R2) + (5*p*q/R2)*C7);
    
        #return np.array([Fx,Fy,self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    #tensile
    def __generateStressZTensile(self):
        self.functionsStressZ = [self.__StressZTensileUA,self.__StressZTensileUB,self.__StressZTensileUC];
    
    def __StressZTensileUA(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        p = self.get_p(y, z);
        q = self.get_q(y, z);
        s = self.get_s(p, q);
        t = self.get_t(p, q);
        d = self.get_d();

        Uprime = self.get_Uprime(x, y, z);
        Wprime = self.get_Wprime(Uprime);
        
        Fx = (1-self.alpha)/2 * (3*x*d/R5) - self.alpha/2 * (3*x*q/R5)*Wprime;
        Fy = -(1-self.alpha)/2 *(1/R3)*(math.sin(2*self.dip) - 3*d*t/R2) - self.alpha/2 * 3*y*q/R5 * Wprime;
        Fz = (1-self.alpha)/2 *(1/R3)*(math.cos(2*self.dip) + 3*d*s/R2) - self.alpha/2 * 3*d*q/R5 *Wprime + self.alpha/2 * 3*q**2/R5;
        
        #return np.array([Fx,2*self.M0*Fy,self.lam/self.mu * self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    
    def __StressZTensileUB(self,x,y,z):
        R3 = self.get_Rx(x,y,z,3);
        R5 = self.get_Rx(x,y,z,5);
        A3 = self.get_A3(x, y, z);
        q = self.get_q(y, z);

        
        K1 = self.get_K1(x, y, z);
        K2 = self.get_K2(x, y, z);
        K3 = self.get_K3(x, y, z, K2);
        
        Uprime = self.get_Uprime(x, y, z);
        Wprime = self.get_Wprime(Uprime);
        
        Fx = 3*x*q/R5*Wprime + (1-self.alpha)/self.alpha * K3*math.sin(self.dip)**2;
        Fy = 3*y*q/R5*Wprime + (1-self.alpha)/self.alpha *K1*math.sin(self.dip)**2;
        Fz = 3*self.c*q/R5 * Wprime - (1-self.alpha)/self.alpha * A3/R3 * math.sin(self.dip)**2;
        
        #return np.array([Fx,2*self.M0*Fy,self.lam/self.mu * self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
    
    def __StressZTensileUC(self,x,y,z):
        R2 = self.get_R2(x,y,z);
        R5 = self.get_Rx(x,y,z,5);
        R7 = self.get_Rx(x,y,z,7);
        A5 = self.get_A5(x, y, z);
        p = self.get_p(y, z);
        q = self.get_q(y, z);
        s = self.get_s(p, q);
        d = self.get_d();
        
        
        B5 = self.get_B5(x, y, z);
        C5 = self.get_C5(x, y, z);
        C7 = self.get_C7(x, y, z);
        
        Fx = -(1-self.alpha)*(3*x/R5)*(math.cos(2*self.dip) + 5*d*s/R2) + self.alpha*(15*self.c*x/R7)*(s-d+7*d*q**2/R2)-self.alpha * (3*x/R5)*(1+(5*d*z/R2));
        Fy = (1-self.alpha)*(3/R5)*(d*B5*math.sin(2*self.dip)-y*C5*math.cos(2*self.dip)) + self.alpha*(3*self.c/R5)*((3+A5)*math.sin(2*self.dip)-(5*y*d/R2)*(2-7*q**2/R2))-self.alpha*(3*y/R5)*(1+5*d*z/R2);
        Fz = -(1-self.alpha)*(3*d/R5)*(1-A5*math.sin(self.dip)**2) - self.alpha*(3*self.c/R5)*(math.cos(2*self.dip)+(10*d*(s-d)/R2)-5*q**2/R2*C7) - self.alpha*(3*z/R5)*(1+C5);
        
        #return np.array([Fx,2*self.M0*Fy,self.lam/self.mu * self.M0*Fz]);
        return np.array([Fx,Fy,Fz]);
        
    
    
    
    
    
    def evaluateDisplacement(self,x,y,z):
        self.set_d(z);
        return self.US*(self.M0/(2*self.mu * math.pi) * (self.functionsDisplacement[0](x,y,z) - self.functionsDisplacement[0](x,y,-z) + self.functionsDisplacement[1](x,y,z)  + z*self.functionsDisplacement[2](x,y,z))) + self.UD * (self.M0/(2*self.mu * math.pi)  *  (self.functionsDisplacement[3](x,y,z) - self.functionsDisplacement[3](x,y,-z) + self.functionsDisplacement[4](x,y,z)  + z*self.functionsDisplacement[5](x,y,z)) );
    
        
    
    def evaluateStressTensor(self,x,y,z):
        if (self.functionsStressX ==None or self.functionsStressY ==None or self.functionsStressZ ==None):
            # print(self.functionsStressX);
            # print(self.functionsStressY);
            # print(self.functionsStressZ);
            raise Exception("ValueError: Functions are not instantiated.");
        self.set_d(z);
        dudr = np.array([self.__evaluateStressX(x,y,z),self.__evaluateStressY(x, y, z),self.__evaluateStressZ(x, y, z)]);
        
        #see page 1024 of the okada pdf
        
        e00 = .5 * (dudr[0][0] + dudr[0][0]);
        e01 = .5 * (dudr[0][1] + dudr[1][0]);
        e02 = .5 * (dudr[0][2] + dudr[2][0]);
    
        e10 = .5 * (dudr[1][0] + dudr[0][1]);
        e11 = .5 * (dudr[1][1] + dudr[1][1]);
        e12 = .5 * (dudr[1][2] + dudr[2][1]);
    
        e20 = .5 * (dudr[2][0] + dudr[0][2]);
        e21 = .5 * (dudr[2][1] + dudr[1][2]);
        e22 = .5 * (dudr[2][2] + dudr[2][2]);
    
        e = np.array( [[e00,e01,e02],[e10,e11,e12],[e20,e21,e22]]);
        
        s00 = self.lam* e[0][0] + 2 * self.mu * e[0][0];
        s01 = 2 * self.mu * e[0][1];
        s02 = 2 * self.mu * e[0][2];
        
        s10 = 2 * self.mu * e[1][0];
        s11 = self.lam* e[1][1] + 2 * self.mu * e[1][1];
        s12 = 2 * self.mu * e[1][2];
        
        s20 = 2 * self.mu * e[2][0];
        s21 = 2 * self.mu * e[2][1];
        s22 = self.lam* e[2][2] + 2 * self.mu * e[2][2];
        
        stressArray = np.array([[s00,s01,s02],[s10,s11,s12],[s20,s21,s22]]);
        
        return stressArray;
    
    def getNormalStress(self, stressArray,otherNode): #normal stress   sigma_ij*ni*nj
        normalVec = self.zAxisRotation(otherNode.normalvector, -(math.pi/2 - self.strikeAngle));
        
        normalStress = np.matmul(np.matmul(stressArray,normalVec),normalVec);
        return normalStress;
    
    def getShearStress(self,stressArray,otherNode):
        normalVec = self.zAxisRotation(otherNode.normalvector, -(math.pi/2 - self.strikeAngle));
        rakeVec = self.zAxisRotation(otherNode.rakeVector, -(math.pi/2 - self.strikeAngle));
        shearStress = np.matmul(np.matmul(stressArray,normalVec),rakeVec);
        return shearStress;
    
    def getStress(self,otherNode): #returns normal and shear stress caused by this node on a different node (othernode)
    
        #apply transforms
        if(otherNode.pointType == "tensile" or otherNode.pointType == "inflation"):
            raise Exception("Stress can't be effected on this node type");
        dist_unrot = self.distance(otherNode = otherNode);
        dist = self.zAxisRotation(dist_unrot, -(math.pi/2 - self.strikeAngle)); #rotate distance vector into coord system
        stressTensor = self.evaluateStressTensor(dist[0],dist[1],dist[2]);
        normalStress = self.getNormalStress(stressTensor, otherNode);
        # #sqrt(T(n)^2 - normal^2)
        # tn2 = np.matmul(np.matmul(stressTensor,normalVec),np.matmul(stressTensor,normalVec));
        # shearStress = np.sqrt(tn2 - normalStress**2);
        shearStress = self.getShearStress(stressTensor,otherNode);
        return np.array([normalStress,shearStress]);
    
    def getStressTensorAt(self,x,y,z):
        dist_unrot = self.distance(dataInput=(True),otherX=x,otherY = y, otherZ = z);
        dist = self.zAxisRotation(dist_unrot, -(math.pi/2 - self.strikeAngle));
        stressTensor = self.evaluateStressTensor(dist[0],dist[1],dist[2]);
        return stressTensor;
    
    def getDisplacementAt(self,x,y,z):
        dist_unrot = self.distance(dataInput=(True),otherX=x,otherY = y, otherZ = z);
        dist = self.zAxisRotation(dist_unrot, -(math.pi/2 - self.strikeAngle));
        dispEval = self.evaluateDisplacement(dist[0], dist[1], dist[2]);
        return dispEval;
        
    
    def pointGen(self,itr,L): #L is the side length of the square being sampled. itr is the power of numbers.
        #Need to adjust so that the source point is in the center of a sampled square. Adjust Z coord by -self.C - L/2
        #adjust X by -L/2
        #do a rotation around the X axis, magnitude of dip angle. X values stay unchanged.
        points = [];
        angle = (math.pi/2 - self.dip);
        RxRot = np.array([[1,0,0],[0,math.cos(angle),-math.sin(angle)],[0,math.sin(angle),math.cos(angle)]]);
        #print(RxRot);
        for i in range(2**(itr)):
            for j in range(2**(itr)):
                tempPoint = np.array([(L * (2*i + 1) * 1/(2**itr)),  0  ,(L * (2*j + 1) * 1/(2**itr))]);
                #print(tempPoint);
                tempPoint[0] = tempPoint[0]-L;
                tempPoint[2] = tempPoint[2]-L;
                #print("mod:",tempPoint); #centered at (0,0) currently
                #print("unrot:",tempPoint);
                #print(RxRot.shape);
                #print(tempPoint.shape);
                
                tempPoint = np.matmul(RxRot,tempPoint);
                #print("rotated");
                #print(tempPoint);
                #print("rot:",tempPoint);
                tempPoint[2] = tempPoint[2]-self.c;
                #print("rotated");
                #print(tempPoint);
                #print("downshift:",tempPoint,"\n");
                points.append(tempPoint);
        #print(points);
        #(RxRot);
        
        return points;

    
    def selfSample(self, sampleDist,samplePower):
    
        normalVec = self.zAxisRotation(self.normalvector, -(math.pi/2 - self.strikeAngle));
        rakeVec = self.zAxisRotation(self.rakeVector, -(math.pi/2 - self.strikeAngle));
        
        #distances = np.array( [[sampleDist,0,-self.c],[-sampleDist,0,0],[0,sampleDist*math.cos(self.dip),-self.c +sampleDist*math.sin(self.dip)],[0,-sampleDist*math.cos(self.dip),-self.c -sampleDist*math.sin(self.dip)]]   ); #picking 4 points on the imagined fault plane. 
        
        #each point will have the same normal and rake vector. Going to evaluate stress tensor on each, find average stresses.
        #print("sampledist:",sampleDist)
        pointItr = samplePower;
        numPoints = 4**(pointItr);
        #stressVec = [None]*numPoints;
        points = self.pointGen(pointItr, sampleDist)
        #print("points");
        #print(points);
        stressVec = np.zeros(shape =(numPoints,3));
        for i in range(len(points)):
            temp = np.matmul(self.evaluateStressTensor(points[i][0],points[i][1],points[i][2]),normalVec)
            stressVec[i][0] = temp[0];
            stressVec[i][1] = temp[1];
            stressVec[i][2] = temp[2]
        
        stressVec = np.average(stressVec,axis = 0);
        
        avgShear = np.dot(stressVec,rakeVec);
        cappingStresses = True;
        if(cappingStresses):
            if(abs(avgShear)>self.shearCapSample):
                if(avgShear<0):
                    avgShear = -self.shearCapSample;
                else:
                    avgShear = self.shearCapSample;
        avgNormal = np.dot(stressVec,normalVec);
        

        return np.array([avgNormal,avgShear]);
    
    def distance(self,otherNode=0,dataInput = False,otherX = False,otherY = False,otherZ = False): #returns a vector of distance between the current node and another, based on the coord system (0,0,-c)
        if(dataInput == False and otherNode == 0):
            raise Exception("Invalid optionset1");
        if(dataInput == True and otherNode != 0):
            raise Exception("Invalid optionset2");
        if(dataInput == False):
            return np.array([otherNode.mapCoordX - self.mapCoordX, otherNode.mapCoordY-self.mapCoordY, (otherNode.mapCoordZ - self.mapCoordZ) + self.c]);
        
        else:
            return np.array([otherX - self.mapCoordX,otherY - self.mapCoordY, (otherZ-self.mapCoordZ)+self.c])
    
    def zAxisRotation(self, vector,angle): #counterclockwise rotation with positive angle
        #3x1 vector
        rotMat = np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]]);
        return np.matmul(rotMat,vector);
        
    
        
