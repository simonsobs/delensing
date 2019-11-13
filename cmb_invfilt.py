# inverse variance filtering

import numpy as np
import healpy as hp
import curvedsky
import prjlib
import pickle
import constants
import cmb
from memory_profiler import profile


def alm_inv(snmin,snmax,fin,fou,cl,nij,itern=2000,eps=1e-6,filter=''):

    lmax = np.size(cl)
    npix = np.size(nij)
    print(np.shape(nij),np.shape(cl))

    for i in range(snmin,snmax):
        Talm = pickle.load(open(fin['T'][i],"rb"))
        Ealm = pickle.load(open(fin['E'][i],"rb"))
        Balm = pickle.load(open(fin['B'][i],"rb"))
        alm  = np.array((Talm,Ealm,Balm))
        print(np.shape(alm),np.shape(nij))
        ilm  = curvedsky.cninv.cnfilterpol(3,npix,lmax,cl,nij,alm,itern,eps=eps,filter=filter)
        pickle.dump((ilm[0,:,:]),open(fou['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump((ilm[1,:,:]),open(fou['E'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump((ilm[2,:,:]),open(fou['B'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    p, f, r = prjlib.analysis_init(t='la',freq='145')
    w, w2, w4 = prjlib.window(f.cmb.mask,p.telescope)
    nij = np.array((w,w,w))
    alm_inv(p.snmin,p.snmax,f.cmb.oalm,f.cmb.ialm,r.lcl[0:3,:],nij)


