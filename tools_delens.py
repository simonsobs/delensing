# Linear template delensing
import numpy as np
import healpy as hp
import pickle
    
# from cmblensplus
import curvedsky
import misctools

# local
import prjlib
import tools_lens


   
def diag_wiener(pqf,clkk,dlmin,dlmax,kL=None,Al=None,klist=['TT','TE','EE','EB']): #kappa filter (including kappa->phi conversion)

    wlk = {}

    for k in klist:

        wlk[k] = np.zeros((dlmax+1))

        if k in ['TT','TE','EE','EB']:

            if Al is None:
                Nl = np.loadtxt(pqf[k].al,unpack=True)[1]
            else:
                Nl = Al[k]

            for l in range(dlmin,dlmax+1):
                wlk[k][l] = clkk[l]/(clkk[l]+Nl[l])
                if kL is not None:
                    wlk[k][l] /= kL[l]

        else:

            wlk[k][dlmin:dlmax+1] = 1.

    return wlk



def template_alm(rlz,qf,elmin,elmax,klmin,klmax,fElm,fdlm,wlk,fgalm='',olmax=2048,glmax=2008,klist=['TT','TE','EE','EB'],**kwargs_ov):

    for k in klist:
    
        for i in rlz:
        
            misctools.progress(i,rlz)
            
            if misctools.check_path(fdlm[k][i],**kwargs_ov): continue
            
            wElm = pickle.load(open(fElm[i].replace('base_maskv3_a5.0deg','base'),"rb"))[:elmax+1,:elmax+1]
            wElm[:elmin,:] = 0.

            if k in ['TT','TE','EE','EB']:
                wplm = wlk[k][:klmax+1,None] * lens_tools.load_klms(qf[k].alm[i],klmax,fmlm=qf[k].mfb[i])
            
            elif k == 'ALLid':
                Glm = np.load(fgalm[i])
                glm = 0.*wElm
                glm[20:glmax+1,:glmax+1] = curvedsky.utils.lm_healpy2healpix(len(Glm),Glm,glmax)[20:,:]
                wplm = glm * wlk[k][:klmax+1,None] #* kL[:dlmax+1,None]
                wplm[:klmin,:] = 0.
           
            dalm = curvedsky.delens.lensingb(olmax,elmin,elmax,klmin,klmax,wElm,wplm)
            pickle.dump((dalm),open(fdlm[k][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

            
            
def template_aps(rlz,fdlm,fBlm,fcl,W,olmax=2048,klist=['TT','TE','EE','EB'],**kwargs_ov):
    
    npix = len(W)
    nside = int(np.sqrt(npix/12.))
    
    for k in klist:
        
        for i in rlz:
        
            if misctools.check_path(fcl[k][i],**kwargs_ov): continue
            if kwargs_ov['verbose']:  misctools.progress(i,rlz)
            
            ii = i - min(rlz)

            dalm = pickle.load(open(fdlm[k][i],"rb"))[0:olmax+1,0:olmax+1]
            wdlm = curvedsky.utils.mulwin_spin(nside,olmax,olmax,2,0*dalm,dalm,W)[1]
            
            Balm = pickle.load(open(fBlm[i],"rb"))[:olmax+1,:olmax+1]
            wBlm = curvedsky.utils.mulwin_spin(nside,olmax,olmax,2,0*Balm,Balm,W)[1]
            
            clbb = curvedsky.utils.alm2cl(olmax,wBlm)
            cldd = curvedsky.utils.alm2cl(olmax,wdlm)
            clbd = curvedsky.utils.alm2cl(olmax,wdlm,wBlm)
            np.savetxt(fcl[k][i],np.array((clbb,cldd,clbd)).T)


            
def cmpute_coeff(rlz,fdlm,fblm,W,olmax=1024,klist=['TT','TE','EE','EB']):

    npix = len(W)
    nside = int(np.sqrt(npix/12.))

    cbb = np.zeros((len(rlz),olmax+1))
    vec = np.zeros((len(rlz),len(klist),olmax+1))
    mat = np.zeros((len(rlz),len(klist),len(klist),olmax+1))
    
    for i in rlz:
        
        ii = i - min(rlz)
        dalm = {}
    
        for k in klist:
            dalm[k] = pickle.load(open(fdlm[k][i],"rb"))[0:olmax+1,0:olmax+1]
            dalm[k] = curvedsky.utils.mulwin_spin(nside,olmax,olmax,2,0*dalm[k],dalm[k],W)[1]
        
        Balm = pickle.load(open(fblm[i],"rb"))[0:olmax+1,0:olmax+1]
        wBlm = curvedsky.utils.mulwin_spin(nside,olmax,olmax,2,0*Balm,Balm,W)[1]
        cbb[ii,:] = curvedsky.utils.alm2cl(olmax,wBlm)
        
        for ki, k0 in enumerate(klist):
            vec[ii,ki,:] = curvedsky.utils.alm2cl(olmax,dalm[k0],wBlm)
            for kj, k1 in enumerate(klist):
                mat[ii,ki,kj,:] = curvedsky.utils.alm2cl(olmax,dalm[k0],dalm[k1])
    
    return np.mean(cbb,axis=0),  np.mean(vec,axis=0),  np.mean(mat,axis=0)



def init_template(dtag,doreal,**kwargs):
    # setup parameters for lensing reconstruction (see cmblensplus/utils/quad_func.py)
    dobj = prjlib.delens(**kwargs)
    prjlib.delens.fname(dobj,dtag,doreal)
    return dobj



def interface(t,fltr,etype,snmin,snmax,ntype,klist,olmax,ow,elmin):

    # //// prepare E modes //// #
    if etype == 'id':
        pE = prjlib.analysis_init(t='id',freq='cocoadd',ntype=ntype)

    if etype == 'co':
        pE = prjlib.analysis_init(t='co',freq='coadd',fltr='cinv',ntype='base')
        #pE = prjlib.analysis_init(t='co',freq='coadd',fltr='cinv',ntype=ntype)

    # //// prepare phi //// #
    if t=='id':
        p = prjlib.analysis_init(t=t,freq='lacoadd',fltr=fltr,snmin=snmin,snmax=snmax,ntype=ntype)
    else:
        p = prjlib.analysis_init(t=t,freq='coadd',fltr=fltr,snmin=snmin,snmax=snmax,ntype=ntype)
    
    # define object
    qobj = tools.lens.init_qobj(p.stag,p.doreal,rlmin=300,rlmax=4096):
    dobj = init_template(p.stag+qobj.ltag+'_'+pE.stag,p.doreal,klist=klist,elmin=elmin)
    
    # pre-filtering for CMB phi
    wlk = diag_wiener(qobj.f,p.kk,dobj.klmin,dobj.klmax,kL=p.kL,klist=klist)

    if fltr == 'cinv':
        if 'TT' in klist:
            wlk['TT'] = 1./p.kL[:dobj.klmax+1]
            qobj.f['TT'].alm = qobj.f['TT'].walm # replaced with kcinv
            qobj.f['TT'].mfb = None
    
    if t == 'id':
        for k in klist:  qobj.f[k].mf = None
    
    # //// compute lensing template alm //// #
    template_alm(p.rlz,qobj.f,dobj.elmin,dobj.elmax,dobj.klmin,dobj.klmax,pE.fcmb.alms['o']['E'],dobj.falm,wlk,fgalm=dobj.fgalm,olmax=olmax,klist=klist,overwrite=ow)

    # //// compute lensing template spectrum projected on SAT area //// #
    # prepare B mode
    Wsa, __ = prjlib.window('sa')
    pid = prjlib.analysis_init(t='id',ntype='cv',snmin=snmin,snmax=snmax)

    # compute aps
    template_aps(p.rlz,dobj.falm,pid.fcmb.alms['o']['B'],dobj.cl,Wsa,olmax=olmax,klist=klist,overwrite=ow) # ignore E-to-B leakage



