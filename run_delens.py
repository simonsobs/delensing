#!/usr/bin/env python
import numpy as np, prjlib, delens_tools


ms = ['ideal']
ms = ['wiener']

for method in ms:
    #for t in ['co','la']:
    for t in ['co']:
        p, f, r = prjlib.analysis_init(t=t,freq='coadd',snmax=100)
        delens_tools.delens_rlz(p.telescope,p.snmin,p.snmax,p.dlmax,p.dlmin,p.dlmax,r.kL,p,f,method,gtype='lss')


