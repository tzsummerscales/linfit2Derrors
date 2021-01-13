# linfit2Derrors
Performs a linear fit on data with errors in both the independent and dependent variable following B. Reed "Linear least-squares fits with errors in both coordinates. II: Comments on parameter variances", Am. J. Phys, 60, 1992

Reference: https://aapt.scitation.org/doi/10.1119/1.17044

The purpose of this project is to perform a linear fit, determining the best slope and y-intercept along with the uncertainties in the slope and y-intercept, when there are errors in both the x and y values of each data point.  The code minimizes a weigted-squared residual sum, implimenting in Python the calculations outlined by B. Reed in his 1992 paper.

## Files
weighted2D.py - contains the functions that perform the linear fit calculation

example.py - a script that runs the fitting functions on some example data
