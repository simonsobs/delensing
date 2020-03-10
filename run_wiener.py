import prjlib, cmb_tools

nside = 1024
t = 'co'

p, f, r = prjlib.analysis_init(t=t,snmin=11,snmax=11)

#cmb_tools.map2alm_wiener_rlz(t,f.cmb,nside,p.snmin,p.snmax,p.dlmax,r.lcl,overwrite=True)

p, f, r = prjlib.analysis_init(t=t,snmin=11,snmax=100)

lmaxs = [p.dlmax,1000,400,200,100,20]
nsides = [nside,512,256,128,64,64]
nsides1 = [512,256,256,128,64,64]
eps = [1e-6,.1,.1,.1,.1,0.]
itns = [100,9,3,3,7,0]
cmb_tools.map2alm_wiener_rlz(t,f.cmb.walm,nside,p.snmin,p.snmax,p.dlmax,r.lcl,overwrite=True,chn=6,lmaxs=lmaxs,nsides=nsides,nsides1=nsides1,itns=itns,eps=eps,reducmn=2)


