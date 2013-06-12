import numpy as np
def LinearLeastSquaresFit(xdata, ydata):
    """
    Take in arrays representing (x,y) values for a set of lienarly varying data and perform a linear
    least squares regression. Return the resulting slope and intercept parameters of the best fit
    line with their uncertainties.
    """
    n = float(xdata.size)
    x = sum(xdata)/n
    y = sum(ydata)/n
    xx = sum(xdata**2)/n
    xy = sum(xdata*ydata)/n
    
    m = (xy - x*y)/(xx - x**2)
    b = (xx*y - x*xy)/(xx - x**2)
    
    delsq = sum((ydata-(m*xdata + b))**2)/n    
    sigy = np.sqrt( (1/(n-2)*delsq) )
    sigm = np.sqrt( (1/(n-2))* (delsq/(xx-x**2)) )
    sigb = np.sqrt( (1/(n-2))* (delsq*xx/(xx-x**2)) )
    return m, b, sigm, sigb

def WeightedLinearLeastSquaresFit(xdata,ydata,wdata):
    """
    Take in arrays representing (xdata,ydata) values for a set of linearly varying
    data and an array of weights wdata. Perform a weighted linear least squares regression.
    Return the resulting slope and intercept parameters of the best fit line with their uncertainties.

    If the weights are all equal to one, the uncertainies on the parameters are calculated using the 
    non-weighted least squares equations.
    """
    n = float(xdata.size)

    if np.all(wdata == 1):
        x = sum(xdata)/n
        y = sum(ydata)/n
        xx = sum(xdata**2)/n
        xy = sum(xdata*ydata)/n
        
        m = (xy - x*y)/(xx - x**2)
        b = (xx*y - x*xy)/(xx - x**2)
    
        delsq = sum((ydata-(m*xdata + b))**2)/n    
        sigy = np.sqrt( (1/(n-2)*delsq) )
        sigm = np.sqrt( (1/(n-2))* (delsq/(xx-x**2)) )
        sigb = np.sqrt( (1/(n-2))* (delsq*xx/(xx-x**2)) )
    else:
        wdata = 1/wdata**2
        w = sum(wdata)/n
        x = sum(wdata*xdata)/n
        y = sum(wdata*ydata)/n
        xx = sum(wdata*xdata**2)/n
        xy = sum(wdata*xdata*ydata)/n
        m = (w*xy - x*y)/(w*xx - x**2)
        b = (xx*y - x*xy)/(w*xx - x**2)
        
        sigm = np.sqrt( (w)/(w*xx - x**2) )
        sigb = np.sqrt( (xx)/(w*xx - x**2) )
        
    return m, b, sigm, sigb