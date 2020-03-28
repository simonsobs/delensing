# Reconstruction
import numpy as np
import curvedsky
import prjlib
import quad_func
import pickle
import basic

def quad_comp_cls(wtype,pq,snmin,snmax,fpalm,fsky,w1,w2,w4,verbose=False):

    for q in pq.qlist:

        cl = np.zeros((snmax-snmin+1,4,pq.oLmax+1))
        #mfg, mfc = pickle.load(open(pq.f[q].mf,"rb"))
        mfg, mfc = 0., 0.

        if wtype=='w':
            if q=='TT':  W2, W4 = w2, w4
            if q in ['TE','TB']:  W2, W4 = w1, w2
            if q in ['EE','EB','MV']:  W2, W4 = fsky, fsky
        else:
            W2, W4 = w2, w4

        for i in range(snmin,snmax+1):

            if verbose: print(i)

            # load reconstructed kappa and curl alms
            glm, clm = pickle.load(open(pq.f[q].alm[i],"rb"))

            # load kappa
            klm = prjlib.load_input_plm(fpalm[i],pq.oLmax,kL=pq.kL)

            # compute cls
            ii = i - snmin
            cl[ii,0,:] = curvedsky.utils.alm2cl(pq.oLmax,glm-mfg)/W4
            cl[ii,1,:] = curvedsky.utils.alm2cl(pq.oLmax,clm-mfc)/W4
            cl[ii,2,:] = curvedsky.utils.alm2cl(pq.oLmax,glm-mfg,klm)/W2
            cl[ii,3,:] = curvedsky.utils.alm2cl(pq.oLmax,klm)
            np.savetxt(pq.f[q].cl[i],np.concatenate((pq.eL[None,:],cl[ii,:,:])).T)

        # save mean
        if snmin!=0 and snmax>=2:
            np.savetxt(pq.f[q].mcls,np.concatenate((pq.eL[None,:],np.mean(cl[1:,:,:],axis=0),np.std(cl[1:,:,:],axis=0))).T)


if __name__ == '__main__':

    #qlist = ['TT','EE','EB']
    #qlist = ['TT','TE','EE','EB']
    qlist = ['TE']
    s = 'W'
    s = 'i'
    s = 'w'
    ow = False

    exttag = ''
    if s=='W': exttag = '_wdiag'
    if s=='i': exttag = '_iso'

    p, f, r = prjlib.analysis_init(t='la',freq='coadd',snmin=1,snmax=100,rlmin=500,rlmax=3000,qlist=qlist,exttag=exttag)
    if s=='i':
        w2, w4 = 1., 1.
    else:
        __, __, fsky, w1, w2, w4 = prjlib.window(f.cmb.amask,p.telescope)

    if s in ['W','i']:
        ocl = prjlib.loadocl(f.cmb.scl['o'],lTmin=p.lTmin,lTmax=p.lTmax)
        ifl = r.lcl[0:3,:]
    if s == 'w':
        ocl, ifl = prjlib.quad_filter(f.cmb.scl,p.lmax,r.lcl,sinp=s,lTmin=p.lTmin,lTmax=p.lTmax)

    quad_func.quad.cinvfilter(p.quad,ocl=ifl)
    quad_func.quad.al(p.quad,r.lcl,ocl,overwrite=ow)
    quad_func.quad.qrec(p.quad,p.snmin,p.snmax,f.cmb.alms[s],r.lcl,overwrite=ow)
    ow = True
    quad_func.quad.n0(p.quad,f.cmb.alms[s],w4,r.lcl,overwrite=ow)
    quad_func.quad.mean(p.quad,w4,overwrite=ow)
    quad_comp_cls(s,p.quad,p.snmin,p.snmax,f.palm,w2,w4,m2,m4,m1w1,m2w2)


