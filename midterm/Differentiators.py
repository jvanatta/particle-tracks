import numpy as np
from numpy import *

def finiteDifference(x,y):
    """
    Differentiats a function defined as two numpy arrays, x and y
    uses a simple center average system with even simpler one direction
    projections for the endpoints
    the input arrays must be the same size
    in: 2 arrays of floats
    out: array of floats
    """
    #quick input checking
    if x.shape != y.shape:
        print "error: array mismatch"
        return NaN
    
    dydxC = zeros(y.shape,float)
    #body case, average difference of the previous and next elements
    dydxC[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    #edge cases, must be handled manually
    dydxC[0] = (y[1] - y[0]) / (x[1] - x[0])
    dydxC[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dydxC


def fourPtFiniteDiff(x,y):
    """
    Differentiats a function defined as two numpy arrays, x and y
    uses a four point differention, more accurate than a simple 
    center average. End points and second-from-endpoints are 
    handled manually with simpler methods
    the input arrays must be the same size
    in: 2 arrays of floats
    out: array of floats
    """
    #simple input checking
    if x.shape != y.shape:
        print "error: array mismatch"
        return NaN
    
    dydx = zeros(y.shape,float)
    #extreme edges
    dydx[0] = (y[1] - y[0]) / (x[1] - x[0])
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    #one in from the edges, use center weighted
    dydx[-2] = (y[-1] - y[-3]) / (2*(x[-1] - x[-2]))
    dydx[1] = (y[2] - y[0]) / (2*(x[-1] - x[-2]))

    #juicy center of the function, use four points, adjacent weighted heavily
    #and once removed from adjacent weighted lightly
    for i, m in enumerate(y[2:-2]):
        #reindex! we arent starting at 0 on this slice
        i+=2
        dydx[i] = (y[i-2] - 8*y[i-1] + 8*y[i+1] - y[i+2]) / (12 * (x[i+1] - x[i]))
    return dydx