#!/usr/bin/env python
import numpy as np, prjlib, basic, delens_tools
from matplotlib.pyplot import *


ms = ['ideal']
for method in ms:
    for t in ['co','la']:
        p, f, r = prjlib.analysis_init(t=t,freq='coadd',snmax=100)
        delens_tools.delens_rlz(p.telescope,p.snmin,p.snmax,p.dlmax,p.dlmin,p.dlmax,r.kL,p,f,method,gtype='lss')


