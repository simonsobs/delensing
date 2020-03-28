#!/usr/bin/env python
import numpy as np, prjlib, cmb_tools

vb = True
ow = False
snmin = 1
snmax = 100

do_map2alm = False
do_combfreq = False
do_cinv_la_diag = False
do_cinv_iso = True
do_cinv_la = False
do_cinv_co = False


if do_map2alm:

    # map -> alm for each freq and telescope
    for freq in ['93','145','225']:
        for t, lmax in [('sa',2048),('la',4096)]: # t should be sa, la or id
            p, f, __ = prjlib.analysis_init(t=t,freq=freq,snmin=snmin,snmax=snmax,lmax=lmax) # define parameters, filenames
            w, __, __, __, w2, __ = prjlib.window(f.cmb.amask,t) # load window function
            cmb_tools.map2alm_rlz(t,p.snmin,p.snmax,freq,p.nside,p.lmax,f.cmb,w,verbose=vb,overwrite=ow) # map -> alm
            cmb_tools.aps(p.snmin,p.snmax,p.lmax,f.cmb,w2,verbose=vb,overwrite=ow) # calculate c

    for t, lmax in [('id',4096)]:
        p, f, __ = prjlib.analysis_init(t=t,snmin=snmin,snmax=snmax,lmax=lmax) # define parameters, filenames
        cmb_tools.map2alm_rlz(t,p.snmin,p.snmax,p.freq,p.nside,p.lmax,f.cmb,1.,verbose=vb,overwrite=ow) # map -> alm
        cmb_tools.aps(p.snmin,p.snmax,p.lmax,f.cmb,1.,stype=['o'],verbose=vb,overwrite=ow) # calculate c


if do_combfreq:
    # combine alm over freq
    for t, lmax, lc in [('sa',2048,1000),('la',4096,4096)]:
        p, f, __ = prjlib.analysis_init(t=t,freq='coadd',snmin=snmin,snmax=snmax,lmax=lmax)
        __, __, __, __, w2, __ = prjlib.window(f.cmb.amask,t) # load window function
        cmb_tools.alm_comb_freq(t,p.snmin,p.snmax,f.cmb,lcut=lc,verbose=vb,overwrite=ow)
        cmb_tools.aps(p.snmin,p.snmax,p.lmax,f.cmb,w2,verbose=vb,overwrite=ow)


if do_cinv_la_diag: 
    # diagonal wiener-filtered alms

    p, f, r = prjlib.analysis_init(t='la',freq='coadd',snmin=snmin,snmax=snmax,lmax=4096)
    __, __, __, __, w2, __ = prjlib.window(f.cmb.amask,t) # load window function

    ocl = prjlib.loadocl(f.cmb.scl['o'],lTmin=p.lTmin,lTmax=p.lTmax)
    cmb_tools.map2alm_wiener_diag(p.snmin,p.snmax,f.cmb.alms['o'],f.cmb.alms['W'],2,p.lmax,r.lcl[0:3,:],ocl[0:3,:],mtype=['T','E','B'],overwrite=ow)
    cmb_tools.aps(p.snmin,p.snmax,p.lmax,f.cmb,w2,stype=['W'],verbose=vb,overwrite=ow)


if do_cinv_iso:
    # diagonal wiener-filtered alms for isotropic noise

    p, f, r = prjlib.analysis_init(t='la',freq='coadd',snmin=snmin,snmax=snmax,lmax=4096)
    ocl = prjlib.loadocl(f.cmb.scl['o'],lTmin=p.lTmin,lTmax=p.lTmax)
    ncl = prjlib.loadocl(f.cmb.scl['n'],lTmin=p.lTmin,lTmax=p.lTmax)
    cmb_tools.wiener_iso(p.snmin,p.snmax,2,p.lmax,f.cmb.alms['i'],r.lcl[0:3,:],ocl[0:3,:],ncl[0:3,:],overwrite=ow)


if do_cinv_la: 
    # full wiener filtering

    p, f, r = prjlib.analysis_init(t='la',freq='coadd',snmin=snmin,snmax=snmax,lmax=4096)
    __, __, fsky, __, w2, __ = prjlib.window(f.cmb.amask,t) # load window function

    # Temperature
    ocl = prjlib.loadocl(f.cmb.scl['o'],lTmin=p.lTmin,lTmax=p.lTmax)
    cmb_tools.map2alm_wiener_diag(p.snmin,p.snmax,f.cmb.alms['o'],f.cmb.alms['w'],2,p.lmax,r.lcl[0:3,:],ocl[0:3,:],mtype=['T'],overwrite=ow)
    cmb_tools.aps(p.snmin,p.snmax,p.lmax,f.cmb,w2,stype=['w'],mtype=['T'],verbose=vb,overwrite=ow)

    # Polarization
    nside = 2048
    eps = [1e-6,.1,.1,.1,.1,0.]
    lmaxs = [p.lmax,2048,1024,512,256,20]
    nsides = [nside,1024,512,256,128,64]
    itns = [100,7,5,3,3,0]
    cmb_tools.map2alm_wiener_rlz(t,f.cmb.alms['w'],nside,p.snmin,p.snmax,p.lmax,r.lcl,overwrite=ow,chn=6,lmaxs=lmaxs,nsides=nsides,itns=itns,eps=eps)

    # aps
    #cmb_tools.map2alm_convert(p.snmin,p.snmax,f.cmb.ialm,f.cmb.walm,p.lmin,p.lmax,r.lcl,overwrite=ow)
    cmb_tools.aps(p.snmin,p.snmax,p.lmax,f.cmb,fsky,stype=['w'],mtype=['E','B'],verbose=vb,overwrite=ow)

    # cross
    __, g, __ = prjlib.analysis_init(t='id',snmin=snmin,snmax=snmax,lmax=p.lmax)
    cmb_tools.apsx(p.snmin,p.snmax,p.lmax,f.cmb,g.cmb,fsky,verbose=vb,overwrite=ow)


if do_cinv_co:

    p, f, r = prjlib.analysis_init(t='co',freq='coadd',snmin=snmin,snmax=snmax)

    nside = 1024
    lmaxs = [p.dlmax,1000,400,200,100,20]
    nsides = [nside,512,256,128,64,64]
    nsides1 = [512,256,256,128,64,64]
    eps = [1e-6,.1,.1,.1,.1,0.]
    itns = [100,9,3,3,7,0]
    cmb_tools.map2alm_wiener_rlz(t,f.cmb.walm,nside,p.snmin,p.snmax,p.dlmax,r.lcl,overwrite=ow,chn=6,lmaxs=lmaxs,nsides=nsides,nsides1=nsides1,itns=itns,eps=eps,reducmn=2)
    __, __, fsky, __, __, __ = prjlib.window(f.cmb.amask,t) # load window function
    cmb_tools.aps(p.snmin,p.snmax,p.lmax,f.cmb,m2,stype=['w'],mtype=['E','B'],verbose=vb,overwrite=ow)


