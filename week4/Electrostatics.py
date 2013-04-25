import numpy as np
from numpy import *
k = 8.987551787368176E-9

def pointPotential(x, y, q, posx, posy):
    """
    calculates the potential at a point or over an array due to a point particle
    in: float(s) x and y, the values to calculate the potential on
       float q, the signed magnitude of the charge
       float posx and posy, the location of the point charge
    out: float(s) of potential values
    """
    return ((k*q) / (((x-posx)**2 + (y-posy)**2))**.5)

def dipolePotential(x, y, q, d, rot):
    """
    calculates the potential at a point or over an array due to a dipole
    evenly spaces two point potentials about the origin
    in: float(s) x and y, the values to calculate the potential on
       float q, the signed magnitude of the charge
       float d, the radial distance separating the dipole
       float rot, the angle of rotation in degrees to rotate the dipole about the orign
           0 degrees is positive pole on the +x axis, CCW rotation
    out: float(s) of potential values
    """
    #calibrate rotation so that 0 is positive dipole along x and the turn is ccw
    rot = -rot + 90
    #convert to radians
    rot = rot * 2* pi/360
    #rotate the positions of the point particles
    xrot = d*sin(rot)
    yrot = d*cos(rot)
    return pointPotential(x, y, q, xrot, yrot) - pointPotential(x, y, q, -xrot, -yrot)
