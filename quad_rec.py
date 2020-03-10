# Reconstruction
import numpy as np
import healpy as hp
import curvedsky
import prjlib
import quad_func
import pickle
import basic

p, f, r = prjlib.analysis_init()
w, w2, w4 = prjlib.window(f.cmb.amask,p.telescope)
ocl = prjlib.loadocl(f.cmb.scl)
ow = True
quad_func.quad.diagcinv(p.quad,ocl)
quad_func.quad.al(p.quad,r.lcl,ocl)
quad_func.quad.qrec(p.quad,p.snmin,p.snmax,f.cmb.oalm,r.lcl,overwrite=ow)
quad_func.quad.n0(p.quad,f.cmb.oalm,w4,r.lcl,overwrite=ow)
quad_func.quad.mean(p.quad,w4,overwrite=ow)

pq = p.quad

for q in pq.qlist:

    cl = np.zeros((p.simn,4,pq.oLmax+1))

    for i in range(p.snmin,p.snmax+1):

        print(i)
        glm, clm = pickle.load(open(pq.f[q].alm[i],"rb"))
        mfg, mfc = pickle.load(open(pq.f[q].mfb[i],"rb"))

        klm = np.complex128(hp.fitsfunc.read_alm(f.palm[i]))
        klm = curvedsky.utils.lm_healpy2healpix(len(klm),klm,5100)[:pq.oLmax+1,:pq.oLmax+1]
        klm *= pq.kL[:,None]

        cl[i,0,:] = curvedsky.utils.alm2cl(pq.oLmax,glm-mfg)/w4
        cl[i,1,:] = curvedsky.utils.alm2cl(pq.oLmax,clm-mfc)/w4
        cl[i,2,:] = curvedsky.utils.alm2cl(pq.oLmax,glm-mfg,klm)/w2
        cl[i,3,:] = curvedsky.utils.alm2cl(pq.oLmax,klm)
        np.savetxt(pq.f[q].cl[i],np.concatenate((pq.eL[None,:],cl[i,:,:])).T)

    # save to file
    if p.snmax>=1:
        print('save sim')
        np.savetxt(pq.f[q].mcls,np.concatenate((pq.eL[None,:],np.mean(cl[1:,:,:],axis=0),np.std(cl[1:,:,:],axis=0))).T)

    if p.snmin==0:
        print('save obs')
        np.savetxt(pq.f[q].ocls,np.concatenate((pq.eL[None,:],cl[0,:,:])).T)

