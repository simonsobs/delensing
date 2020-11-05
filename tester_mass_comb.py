#!/usr/bin/env python
# coding: utf-8

import numpy as np, prjlib, healpy as hp, pickle, curvedsky as cs, tools_delens, tools_lens, tools_multitracer, tqdm
import warnings
warnings.filterwarnings("ignore")

olmax = 2048
elmin, elmax = 50, 2048
klmin, klmax = 20, 2048

pE = prjlib.analysis_init(t='co',freq='com',fltr='cinv',ntype='base_roll50')
pid = prjlib.analysis_init(t='id',ntype='cv')

W = prjlib.window('la',ascale=5.,nside=2048)[0]
M = prjlib.window('la',ascale=0.,nside=2048)[0]

Wsa = prjlib.window('sa')[0]
#Wsa *= hp.pixelfunc.ud_grade(M,512)
Wsa *= hp.pixelfunc.ud_grade(W,512)

ntypes = ['w','iso_m']

for n in ntypes:
    
    if 'iso' in n:
        pobj = prjlib.analysis_init(t='la',freq='com',fltr='none',snmin=1,snmax=100,ntype='base_iso_roll50')
    else:
        pobj = prjlib.analysis_init(t='la',freq='com',fltr='none',snmin=1,snmax=100,ntype='base_roll50')
    qobj = tools_lens.init_qobj(pobj.stag,False,rlmin=300,rlmax=4096,qlist=['TT','TE','EE','EB'])
    mobj = tools_multitracer.mass_tracer( pobj, qobj, add_cmb = ['TT','TE','EE','EB'] )
    
    rho = 0.
    
    wlk = tools_delens.diag_wiener( qobj.f, pobj.kk, klmin, klmax, kL=pobj.kL, klist=['comb'] )
    
    signal_covariance, clnl_matrix = tools_multitracer.get_spectra_matrix( mobj ) # for analytic filter
    weight = tools_multitracer.calculate_multitracer_weights( clnl_matrix, signal_covariance[0,0,:], mobj.lmin )

    Al = (np.loadtxt(qobj.f['TT'].al)).T[1]
    wfac = np.sum(Al[:klmax+1]*(2*np.linspace(0,klmax,klmax+1)+1)/4./np.pi)
    if 'iso' not in n: 
        nkk  = pickle.load(open(qobj.f['TT'].nkmap,"rb")) / wfac

    if n=='m':  mmask, kmask = M, None
    if n=='w':  mmask, kmask = W, None
    if n=='I':  mmask, kmask = M, M/(nkk**0.5+1e-30)
    if n=='iso_m':  mmask, kmask = M, M
    if n=='iso_w':  mmask, kmask = W, M
    if n=='iso_wm':  mmask, kmask = M, W
    if n=='iso_ww':  mmask, kmask = W, W

    for i in tqdm.tqdm(pobj.rlz):
        wElm = pickle.load(open(pE.fcmb.alms['o']['E'][i],"rb"))[:elmax+1,:elmax+1]
        iBlm = pickle.load(open(pid.fcmb.alms['o']['B'][i],"rb"))[:olmax+1,:olmax+1]
        wBlm = cs.utils.mulwin_spin( 0*iBlm, iBlm, Wsa )[1]
        alm = tools_multitracer.load_mass_tracers( i, qobj, mobj, mmask=mmask, kmask = kmask)
        klm = 0.*wElm
        klm[:2008,:2008] = tools_multitracer.coadd_kappa_alms( alm, weight )
        wplm = klm[:klmax+1,:klmax+1] * wlk['comb'][:klmax+1,None]
        dalm = cs.delens.lensingb( olmax, elmin, elmax, klmin, klmax, wElm, wplm )
        wdlm = cs.utils.mulwin_spin( 0*dalm, dalm, Wsa )[1]
        rho += cs.utils.alm2rho(olmax,wdlm,wBlm)/len(pobj.rlz)

    np.savetxt('misc/test_mass_narrow_'+n+'.dat',rho.T)

