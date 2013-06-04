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