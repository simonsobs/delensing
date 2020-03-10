#!/usr/bin/env python
import numpy as np, prjlib, cmb_tools


# map -> alm for each freq and telescope
for freq in ['93','145']:
    for t in ['sa','la','id']: # t should be sa, la or id
        p, f, __ = prjlib.analysis_init(t=t,freq=freq,snmax=100) # define parameters, filenames
        w, w2, w4 = prjlib.window(f.cmb.amask,t) # load window function
        cmb_tools.map2alm_rlz(t,p.snmin,p.snmax,freq,p.nside,p.lmax,f.cmb,w,verbose=False) # map -> alm
        cmb_tools.aps(t,p.snmin,p.snmax,p.lmax,f.cmb,w,w2,w4,verbose=False) # calculate c


# combine alm over freq
for t, lc in [('sa',1000),('la',3000)]:
    p, f, __ = prjlib.analysis_init(t=t,freq='coadd',snmax=10)
    w, w2, w4 = prjlib.window(f.cmb.amask,t) # load window function
    cmb_tools.alm_comb_freq(t,p.snmin,p.snmax,f.cmb,lcut=lc,freqs=freqs,verbose=False)
    cmb_tools.aps(t,p.snmin,p.snmax,p.lmax,f.cmb,w,w2,w4,verbose=False)


# combine alm over freq (and telecsope) with a non-diagonal wiener filtering
p, f, r = prjlib.analysis_init(t='la',snmin=1,snmax=10,lmax=4096)
lmaxs = [p.lmax,2048,1024,512,256,128,20]
nsides = [2048,1024,512,256,128,64,64]
itns = [100,7,5,5,3,3,0]
eps = [1e-6,.01,.1,.1,.1,.1,0.]
cmb_tools.map2alm_wiener_rlz('la',f.cmb.ialm,2048,p.snmin,p.snmax,p.lmax,r.lcl,overwrite=True,chn=7,lmaxs=lmaxs,nsides=nsides,itns=itns,eps=eps)


