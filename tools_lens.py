# from external
import numpy as np
import healpy as hp
import sys
import pickle

# from cmblensplus/wrap/
import curvedsky

# from cmblensplus/utils/
import misctools
import quad_func
import cmb as CMB

# local
import prjlib



def load_klms(falm,lmax,fmlm=None):

    klm = pickle.load(open(falm,"rb"))[0][:lmax+1,:lmax+1]        
    if fmlm is not None:
        mlm = pickle.load(open(fmlm,"rb"))[0][:lmax+1,:lmax+1]
        klm -= mlm
    return klm


def klm_debiased(qobjf,i,lmax):

    klm = pickle.load(open(qobjf.alm[i],"rb"))[0][:lmax+1,:lmax+1]
    mlm = pickle.load(open(qobjf.mfb[i],"rb"))[0][:lmax+1,:lmax+1]
    klm -= mlm

    return klm


def compute_knoise(snmin,snmax,qobjf,W,fpalm,verbose=True,nside=1024,qlist=['TT'],ep=1e-30):
    # used for compute_kcninv

    lmax = 2*nside
    
    w = hp.pixelfunc.ud_grade(W,nside)
    m = w/(w+ep)
    
    rlz = np.linspace(snmin,snmax,snmax-snmin+1,dtype=np.int)
    nkaps, inkaps = {}, {}
    
    nkap = 0.
        
    for i in rlz:

        if verbose:  misctools.progress(i,rlz,addtext='knoise (nside='+str(nside)+')')

        klm  = klm_debiased(qobjf,i,lmax)
        iklm = prjlib.load_input_plm(fpalm[i],lmax,ktype='k')
        iklm = curvedsky.utils.mulwin(nside,lmax,lmax,iklm,w**2)

        kap  = m/(w+ep)**2 * curvedsky.utils.hp_alm2map(nside,lmax,lmax,klm-iklm)
        nkap += m*kap**2/len(rlz)
        
    inkap = m/(nkap+ep)

    pickle.dump((nkap,inkap),open(qobjf.nkmap,"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    
    return nkap, inkap



def compute_kcninv(qobjf,snmin,snmax,ckk,fpalm,nside=1024,Snmin=1,Snmax=100,chn=1,eps=[1e-6],itns=[100],lmaxs=[0],nsides=[0],qlist=['TT','TE'],**kwargs_ov):

    npix = 12*nside**2
    lmax = np.size(ckk) - 1

    rlz = np.linspace(snmin,snmax,snmax-snmin+1,dtype=np.int)

    M, __ = prjlib.window('la',ascale=0.)
    iW2 = M/(W+1e-30)**2

    bl = np.ones((1,lmax+1))
    cl = np.reshape(ckk,(1,lmax+1))

    for q in qlist:

        if misctools.check_path(qobjf[q].nkmap,overwrite=overwrite,verbose=verbose):
            __, inkk = pickle.load(open(qobjf[q].nkmap,"rb"))
        else:
            __, inkk = compute_knoise(Snmin,Snmax,qobjf[q],W,fpalm,nside=nside)

        iNkk = np.reshape(inkk,(1,1,npix))

        for i in rlz:

            if verbose:  misctools.progress(i,rlz,addtext='kcninv ('+q+')')

            kap = np.zeros((1,1,npix))
            klm = klm_debiased(qobjf[q],i,lmax)
            kap[0,0,:] = iW2 * curvedsky.utils.hp_alm2map(nside,lmax,lmax,klm)

            wklm = curvedsky.cninv.cnfilter_freq(1,1,nside,lmax,cl,bl,iNkk,kap,chn,lmaxs=lmaxs,nsides=nsides,itns=itns,eps=eps,filter='w',ro=1)
            pickle.dump((wklm),open(qobjf[q].walm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def aps(fltr,qobj,rlz,fpalm,wn,verbose=True):
    # Compute aps of reconstructed lensing map
    # This code can be used for checking reconstructed map

    for q in qobj.qlist:

        cl = np.zeros((len(rlz),4,qobj.olmax+1))
        #mfg, mfc = pickle.load(open(qobj.f[q].mf,"rb"))

        if fltr == 'cinv':
            if q=='TT':  W2, W4 = wn[2], wn[4]
            if q in ['TE','TB']:  W2, W4 = wn[1], wn[2]
            if q in ['EE','EB','MV']:  W2, W4 = wn[0], wn[0]
        else:
            W2, W4 = wn[2], wn[4]

        for i in rlz:

            if verbose:  print(i)

            # load reconstructed kappa and curl alms
            glm, clm = pickle.load(open(qobj.f[q].alm[i],"rb"))
            mfg, mfc = pickle.load(open(qobj.f[q].mfb[i],"rb"))

            # load kappa
            klm = prjlib.load_input_plm(fpalm[i],qobj.olmax,ktype='k')

            # compute cls
            ii = i - min(rlz)
            cl[ii,0,:] = curvedsky.utils.alm2cl(qobj.olmax,glm-mfg)/W4
            cl[ii,1,:] = curvedsky.utils.alm2cl(qobj.olmax,clm-mfc)/W4
            cl[ii,2,:] = curvedsky.utils.alm2cl(qobj.olmax,glm-mfg,klm)/W2
            cl[ii,3,:] = curvedsky.utils.alm2cl(qobj.olmax,klm)
            np.savetxt(qobj.f[q].cl[i],np.concatenate((qobj.l[None,:],cl[ii,:,:])).T)

        # save sim mean
        if rlz[0]>=1 and len(rlz)>1:
            np.savetxt(qobj.f[q].mcls,np.concatenate((qobj.l[None,:],np.mean(cl[1:,:,:],axis=0),np.std(cl[1:,:,:],axis=0))).T)


def interface(run=[],kwargs_ov={},kwargs_cmb={},kwargs_qrec={},ep=1e-30):

    if kwargs_cmb['t'] != 'la':
        sys.exit('only la is supported')

    # Define parameters, filenames for input CMB
    p = prjlib.analysis_init(**kwargs_cmb)

    # Load pre-computed w-factor which is used for correction normalization of spectrum
    wn = prjlib.wfac(p.telescope)

    # Compute filtering
    if p.fltr == '': # for none-filtered alm
        # Load "observed" aps containing signal, noise, and some residual. 
        # This aps will be used for normalization calculation
        ocl = prjlib.loadocl(p.fcmb.scl['o'],lTmin=p.lTmin,lTmax=p.lTmax)
        # CMB alm will be multiplied by 1/ifl before reconstruction process
        ifl = ocl#p.lcl[0:3,:]

    if p.fltr == 'cinv': # for C^-1 wiener-filtered alm
        pc = prjlib.analysis_init(t=p.telescope,freq='coadd',fltr='',ntype=p.ntype)
        # Compute aps appropriate for C^-1 filtering case. 
        #ocl, ifl = prjlib.quad_filter(p.fcmb.scl,pc.fcmb.scl,p.lmax,p.lcl,lTmin=p.lTmin,lTmax=p.lTmax)
        ocl = np.zeros((3,p.lmax+1))
        ifl = np.zeros((3,p.lmax+1))
        bl  = 1./CMB.beam(1.,p.lmax)
        sig = np.array([10.,14.,14.])
        for teb in range(3):
            cnl = p.lcl[teb,:] + (1./bl)**2*(sig[teb]*np.pi/10800./2.72e6)**2
            fcl = np.loadtxt(p.fcmb.scl['o'],unpack=True)[teb+1]
            # quality factor defined in Planck 2015 lensing paper
            Ql  = (p.lcl[teb,:])**2/(cnl*fcl+ep**2)
            ocl[teb,:] = cnl/(Ql+ep)  # corrected observed cl
            ifl[teb,:] = p.lcl[teb,:]/(Ql+ep)    # remove theory signal cl in wiener filter
        wn[:] = wn[0]

    d = prjlib.data_directory()
    ids = prjlib.rlz_index(doreal=p.doreal)
    qobj = quad_func.reconstruction(d['root'],ids,rlz=p.rlz,stag=p.stag,run=run,wn=wn,lcl=p.lcl,ocl=ocl,ifl=ifl,falm=p.fcmb.alms['o'],**kwargs_ov,**kwargs_qrec)

    # Aps of reconstructed phi
    if 'aps' in run: 
        aps(p.fltr,qobj,p.rlz,p.fpalm,wn)



def init_qobj(stag,doreal,**kwargs):
    # setup parameters for lensing reconstruction (see cmblensplus/utils/quad_func.py)
    d = prjlib.data_directory()
    ids = prjlib.rlz_index(doreal=doreal)
    qobj = quad_func.reconstruction(d['root'],ids,stag=stag,run=[],**kwargs)
    return qobj



