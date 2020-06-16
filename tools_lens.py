# from external
import numpy as np
import healpy as hp
import sys
import pickle
import tqdm

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


def compute_knoise(rlz,qobjf,W,M,iW2,fpalm,lmax,verbose=True,qlist=['TT'],lmin=10,ep=1e-40):
    # used for compute_kcninv

    nside = hp.pixelfunc.get_nside(W)
    nkap  = 0.
        
    for i in tqdm.tqdm(rlz,ncols=100,desc='knoise',leave=False):

        rklm = klm_debiased(qobjf,i,lmax)
        rklm[:lmin,:] = 0.
        rkap = curvedsky.utils.hp_alm2map(nside,lmax,lmax,rklm)

        iklm = prjlib.load_input_plm(fpalm[i],lmax,ktype='k')
        iklm = curvedsky.utils.mulwin(nside,lmax,lmax,iklm,W**2)
        iklm[:lmin,:] = 0.
        ikap = curvedsky.utils.hp_alm2map(nside,lmax,lmax,iklm)

        nkap += iW2**2 * (rkap-ikap)**2/len(rlz)
    
    #inkap = M/(nkap+ep)*(lmax-lmin)*(lmin+lmax+2.)/(4*np.pi)
    inkap = M/(nkap+ep)/(4*np.pi)
        
    pickle.dump((inkap),open(qobjf.nkmap,"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    
    return inkap



def compute_kcninv(qobjf,rlz,fltr,ckk,fpalm,nside=2048,Snmin=1,Snmax=100,klmin=10,chn=1,eps=[1e-5],itns=[100],lmaxs=[0],nsides=[0],qlist=['TT'],**kwargs_ov):

    npix = 12*nside**2
    lmax = np.size(ckk) - 1

    M, __ = prjlib.window('la',ascale=0.)
    bl = np.ones((1,lmax+1))
    cl = np.reshape(ckk,(1,lmax+1))

    #for q in tqdm.tqdm(qlist,ncols=100,desc='kcinv'):
    for q in tqdm.tqdm(['TT'],ncols=100,desc='kcinv'):

        if fltr=='cinv' and q!='TT':
            W = M
            iW2 = 1.
        else:
            W, __ = prjlib.window('la',ascale=5.)
            iW2 = 1./(W**2+1e-60)

        #if misctools.check_path(qobjf[q].nkmap,**kwargs_ov):
        #    inkk = pickle.load(open(qobjf[q].nkmap,"rb"))
        #else:
        #Rlz = np.linspace(Snmin,Snmax,Snmax-Snmin+1,dtype=np.int)
        #inkk = compute_knoise(Rlz,qobjf[q],W,M,iW2,fpalm,lmax,lmin=klmin)
        
        inkk = pickle.load(open(qobjf[q].nkmap,"rb"))
        
        iNkk = np.reshape(inkk,(1,1,npix))
        
        Al = np.loadtxt(qobjf[q].al,unpack=True)[1]
        iNkk = np.mean(1./Al[2:1000]) * iNkk/np.max(iNkk)

        for i in tqdm.tqdm(rlz,ncols=100,desc='each rlz ('+q+'):',leave=False):

            klm = klm_debiased(qobjf[q],i,lmax)
            klm[:klmin,:] = 0.
            kap = np.reshape( M*iW2 * curvedsky.utils.hp_alm2map(nside,lmax,lmax,klm) , (1,1,npix) )

            wklm = curvedsky.cninv.cnfilter_freq(1,1,nside,lmax,cl,bl,iNkk,kap,chn,lmaxs=lmaxs,nsides=nsides,itns=itns,eps=eps,filter='w',ro=1)
            pickle.dump((wklm),open(qobjf[q].walm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def aps(fltr,qobj,rlz,fpalm,wn,verbose=True):
    # Compute aps of reconstructed lensing map
    # This code can be used for checking reconstructed map

    for q in tqdm.tqdm(qobj.qlist,ncols=100,desc='aps'):

        cl = np.zeros((len(rlz),4,qobj.olmax+1))

        W2, W4 = wn[2], wn[4]

        for ii, i in enumerate(tqdm.tqdm(rlz,ncols=100,desc='each rlz ('+q+'):')):

            # load reconstructed kappa and curl alms
            glm, clm = pickle.load(open(qobj.f[q].alm[i],"rb"))
            mfg, mfc = pickle.load(open(qobj.f[q].mfb[i],"rb"))

            # load kappa
            klm = prjlib.load_input_plm(fpalm[i],qobj.olmax,ktype='k')

            # compute cls
            cl[ii,0,:] = curvedsky.utils.alm2cl(qobj.olmax,glm-mfg)/W4
            cl[ii,1,:] = curvedsky.utils.alm2cl(qobj.olmax,clm-mfc)/W4
            cl[ii,2,:] = curvedsky.utils.alm2cl(qobj.olmax,glm-mfg,klm)/W2
            cl[ii,3,:] = curvedsky.utils.alm2cl(qobj.olmax,klm)
            np.savetxt(qobj.f[q].cl[i],np.concatenate((qobj.l[None,:],cl[ii,:,:])).T)

        # save sim mean
        if rlz[0]>=1 and len(rlz)>1:
            np.savetxt(qobj.f[q].mcls,np.concatenate((qobj.l[None,:],np.mean(cl[1:,:,:],axis=0),np.std(cl[1:,:,:],axis=0))).T)


def quad_filter(fcinv,fdiag,lmax,lcl,**kwargs):
    # CMB filtering

    fl = prjlib.loadocl(fcinv['o'],**kwargs)
    ol = prjlib.loadocl(fdiag['o'],**kwargs)
    xl = prjlib.loadocl(fcinv['x'],**kwargs)

    alp = np.zeros((3,lmax+1))
    ocl = np.zeros((4,lmax+1))
    ifl = np.zeros((3,lmax+1))

    alp[1:3,2:] = xl[1:3,2:]/lcl[1:3,2:]

    ocl[0,:] = ol[0,:]
    ocl[1:3,2:] = fl[1:3,2:]/alp[1:3,2:]**2
    ocl[3,:] = ol[3,:]
    ifl[0,:] = lcl[0,:]
    ifl[1:3,2:] = fl[1:3,2:]/alp[1:3,2:]

    return ocl, ifl



def init_qobj(stag,doreal,**kwargs):
    # setup parameters for lensing reconstruction (see cmblensplus/utils/quad_func.py)
    d = prjlib.data_directory()
    ids = prjlib.rlz_index(doreal=doreal)
    qobj = quad_func.reconstruction(d['root'],ids,stag=stag,run=[],**kwargs)
    return qobj



def interface(run=[],kwargs_ov={},kwargs_cmb={},kwargs_qrec={},ep=1e-30):

    if kwargs_cmb['t'] != 'la':
        sys.exit('only la is supported')

    # Define parameters, filenames for input CMB
    p = prjlib.analysis_init(**kwargs_cmb)

    # Load pre-computed w-factor which is used for correction normalization of spectrum
    wn = prjlib.wfac(p.telescope)

    # Compute filtering
    if p.fltr == 'none': # for none-filtered alm
        # Load "observed" aps containing signal, noise, and some residual. 
        # This aps will be used for normalization calculation
        ocl = prjlib.loadocl(p.fcmb.scl['o'],lTmin=p.lTmin,lTmax=p.lTmax)
        # CMB alm will be multiplied by 1/ifl before reconstruction process
        ifl = ocl#p.lcl[0:3,:]

    elif p.fltr == 'cinv': # for C^-1 wiener-filtered alm
        pc = prjlib.analysis_init(t=p.telescope,freq='com',fltr='none',ntype=p.ntype)
        
        # Compute aps appropriate for C^-1 filtering case. 
        ocl, ifl = quad_filter(p.fcmb.scl,pc.fcmb.scl,p.lmax,p.lcl,lTmin=p.lTmin,lTmax=p.lTmax)
        ocl[ocl<=0.] = 1e30
        ifl[ifl<=0.] = 1e30
        wn[:] = wn[0]

    else:
        sys.exit('unknown filtering')

    if 'iso' in p.ntype: #fullsky case
        wn[:] = 1.

    d = prjlib.data_directory()
    ids = prjlib.rlz_index(doreal=p.doreal)
    qobj = quad_func.reconstruction(d['root'],ids,rlz=p.rlz,stag=p.stag,run=run,wn=wn,lcl=p.lcl,ocl=ocl,ifl=ifl,falm=p.fcmb.alms['o'],**kwargs_ov,**kwargs_qrec)

    # Aps of reconstructed phi
    if 'aps' in run: 
        aps(p.fltr,qobj,p.rlz,p.fpalm,wn)

    # Cinv kappa
    if 'kcinv' in run:
        compute_kcninv(qobj.f,p.rlz,p.fltr,p.kk[:qobj.olmax+1],p.fpalm,Snmin=p.snmin,Snmax=p.snmax,qlist=qobj.qlist,**kwargs_ov)



