# -*- coding: utf-8 -*-
"""

weighted2D.py
Contains functions to perform weighted least squares linear fit when there
is error in both the x and y directions

Reference: B. Reed "Linear least-squares fits with errors in both
coordinates. II: Comments on parameter variances", Am. J. Phys, 60, 1992

"""

def mfunc(m, x_in, y_in):
   # MFUNC - function to be minimized in order to find best slope

   import numpy as np

   # Separate x and y from their weights
   x = x_in[:,0]
   y = y_in[:,0]
   Wxi = x_in[:,1]
   Wyi = y_in[:,1]

   # Calculate weight for each data point
   Wi = Wxi*Wyi/(m**2*Wyi+Wxi) # Eq 8

   # Weighted means and deviations from weighted means
   xbar = np.sum(Wi*x)/np.sum(Wi) # Eq 11
   ybar = np.sum(Wi*y)/np.sum(Wi) # Eq 12
   U = x-xbar # Eq 9
   V = y-ybar # Eq 10

   # Minimization function (eq 19 from paper)
   g = (m**2*np.sum((Wi**2)*U*V/Wxi) + m*np.sum((Wi**2)*((U**2)/Wyi - 
   (V**2)/Wxi)) - np.sum((Wi**2)*U*V/Wyi)) 
   g = g**2

   return g
   

def wls2d(x, y, delx, dely):
   """ 
   WLS2D Calculates the weighted least squares fit to a straight line when
   there are errors in both the x and y directions.

   Reference: B. Reed "Linear least-squares fits with errors in both
   coordinates. II: Comments on parameter variances", Am. J. Phys, 60, 1992

   fitparams = wls2d(x, y, delx, dely, flag);

   INPUTS
   x      vector of independent data points
   y      vector of dependent data points
   delx   vector of uncertainties/errors in x points
   dely   vector of uncertainties/errors in y points

   OUTPUT
   fitparams vector of fit parameters
   fitparams[0]   best fit slope
   fitparams[1]   best fit y intercept
   fitparams[2]   uncertainty in slope
   fitparams[3]   uncertainty in y-intercept

   Note: equation numbers from B. Reed's paper
   """

   import numpy as np
   from numpy.matlib import repmat
   from scipy.optimize import fmin

   N = len(x)

   # Calculate weights and weighted means
   Wxi = 1/(delx**2)
   Wyi = 1/(dely**2)
   
   # Force vectors to be column vectors 
   x.shape = (N,1)
   y.shape = (N,1)
   Wxi.shape = (N,1)
   Wyi.shape = (N,1)

   # Add weights as second columns to x and y
   xWxi = np.append(x, Wxi, axis=1)
   yWyi = np.append(y, Wyi, axis=1)

   # Use unweighted linear regression to find a slope initial guess
   m0 = ((N*np.sum(x*y) - np.sum(x)*np.sum(y))/(N*np.sum(x**2) - np.sum(x)**2))

   # Find best slope
   m = fmin(func=mfunc, x0=m0, args=(xWxi, yWyi,))


   # Calculate final weight for each data point
   Wi = Wxi*Wyi/(m**2*Wyi+Wxi) # Eq 8
   Wj = Wi

   # Weighted means & deviations from weighted means
   xbar = np.sum(Wi*x)/np.sum(Wi) # Eq 11
   ybar = np.sum(Wi*y)/np.sum(Wi) # Eq 12
   U = x-xbar # Eq 9
   V = y-ybar # Eq 10

   # Calculate corresponding y-intercept (equation 13)
   c = ybar - m*xbar # Eq 13

   # Sum of weighted residuals
   S = np.sum(Wi*((V-m*U)**2)) # Eq 14

   # Use calculated data points
   lam = Wi*(c + m*x - y) # Eq 26
   x = x - lam*m/Wxi # Eq 24
   y = y + lam/Wyi  # Eq 25
   xbar = np.sum(Wi*x)/np.sum(Wi) # Eq 11
   ybar = np.sum(Wi*y)/np.sum(Wi) # Eq 12 
   U = x-xbar # Eq 9
   V = y-ybar # Eq 10


   # Calculate parameter derivatives (Appendix)
   W = np.sum(Wi) # Eq A10
   HH = -2*m/W*np.sum(Wi**2*V/Wxi) # Eq A11
   JJ = -2*m/W*np.sum(Wi**2*U/Wxi) # Eq A12
   AA = 4*m*np.sum(Wi**3*U*V/Wxi**2) - W*HH*JJ/m # Eq A3
   BB = -np.sum(Wi**2*(4*m*Wi/Wxi*(U**2/Wyi - V**2/Wxi) - 2*V*HH/Wxi + 
   2*U*JJ/Wyi)) # Eq A4
   CC = -np.sum(Wi**2/Wyi*(4*m*Wi*U*V/Wxi + V*JJ + U*HH)) # Eq A5
   delta = np.eye(N)  # Kroneker Delta
   delmat = delta - repmat(Wj,1,N)/W
   DD = np.dot(delmat,(Wi**2*V/Wxi)) # Eq A6
   EE = 2*np.dot(delmat,(Wi**2*U/Wyi)) # Eq A7
   FF = np.dot(delmat,(Wi**2*V/Wyi)) # Eq A8
   GG = np.dot(delmat,(Wi**2*U/Wxi)) # Eq A9
   A = np.sum(Wi**2*U*V/Wxi) # Eq 19 & 20
   B = np.sum(Wi**2*(U**2/Wyi - V**2/Wxi)) # Eq 19 & 20
   dmdxj = -1*(m**2*DD + m*EE - FF)/(2*m*A + B - AA*m**2 + BB*m - CC) # Eq A1
   dmdyj = -1*(m**2*GG - 2*m*DD - 0.5*EE)/(2*m*A + B - AA*m**2 + BB*m - 
   CC); # Eq A2 
   dcdxj = (HH - m*JJ - xbar)*dmdxj - m*Wj/W # Eq A13
   dcdyj = (HH - m*JJ - xbar)*dmdyj + Wj/W # Eq A14
   delm = np.sqrt(S/(N-2)*np.sum(1/Wyi*dmdyj**2 + 1/Wxi*dmdxj**2)) # Eq 21
   delc = np.sqrt(S/(N-2)*np.sum(1/Wyi*dcdyj**2 + 1/Wxi*dcdxj**2)) # Eq 21

   fitparams = np.concatenate((m, c))
   fitparams = np.append(fitparams, delm)
   fitparams = np.append(fitparams, delc)

   return fitparams

