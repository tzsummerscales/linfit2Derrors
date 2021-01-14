# -*- coding: utf-8 -*-
"""

example.py

Performs an example weighted linear fit using data with errors in both
the x and y directions.

"""

import numpy as np
from weighted2D import wls2d

help(wls2d)

xdat = np.array([2.02, 4.01, 6.00, 8.02, 10.01])*1e-3
ydat = np.array([2.02, 3.99, 5.98, 7.99, 9.96])

xerr = xdat*0.008
yerr = ydat*0.005

fitparams = wls2d(xdat,ydat,xerr,yerr)

m = fitparams[0]
c = fitparams[1]
delm = fitparams[2]
delc = fitparams[3]

print('slope = {:.2f}'.format(m))
print('y-int = {:.3f}'.format(c))
print('slope err = {:.3f}'.format(delm))
print('y-int err = {:.3f}'.format(delc))