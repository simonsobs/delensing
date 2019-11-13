#* Linear template delensing

# load modules
import numpy as np
import basic
import pickle
import curvedsky


# noise spectra
def gen_noise_spec(lmax,rlTmax=3000,dd='../../data/forecast/',deproj=0,sens=2,Fsky='0p4',Tcmb=2.726e6):
    nl  = np.zeros((4,lmax+1))
    if sens==1:
        tag = 'baseline'
    if sens==2:
        tag = 'goal'
    fT = dd+'/so/noise/SO_LAT_Nell_T_'+tag+'_fsky'+Fsky+'_ILC_CMB.txt'
    fE = dd+'/so/noise/SO_LAT_Nell_P_'+tag+'_fsky'+Fsky+'_ILC_CMB_E.txt'
    fB = dd+'/so/noise/SO_LAT_Nell_P_'+tag+'_fsky'+Fsky+'_ILC_CMB_B.txt'
    nl[0,40:] = np.loadtxt(fT,unpack=True)[deproj+1,:lmax-39]
    nl[1,10:] = np.loadtxt(fE,unpack=True)[deproj+1,:lmax-9]
    nl[2,10:] = np.loadtxt(fB,unpack=True)[deproj+1,:lmax-9]
    nl[0,rlTmax+1:] = 1e10*Tcmb**2
    return nl/Tcmb**2


D = '../../data/sodelens/20190707_test/'

# define parameters
Tcmb = 2.726e6    # CMB temperature
clmax = 5000
lmax = 4000       # maximum multipole of output cl
rlmin = 300
rlmax = 4000      # reconstruction multipole range
dlmin = 2
dlmax = 2048 
nside = 4096
npix = 12*nside**2
mcnum = 1
phirandom = True # reconstructed phi = input phi + random noise
phirandom = False
qlist = ['TT','EE','EB','MV']

# load unlensed, lensed, noise Cls
ucl = basic.aps.read_cambcls('../../data/cls/ffp10_scalCls.dat',2,clmax,5)/Tcmb**2
lcl = basic.aps.read_cambcls('../../data/cls/ffp10_lensedCls.dat',2,clmax,4,bb=True)/Tcmb**2
nl  = gen_noise_spec(lmax)
ocl = lcl[:,:lmax+1] + nl

cls = np.zeros((mcnum,7,lmax+1))
rls = np.zeros((mcnum,len(qlist),lmax+1))
dls = np.zeros((mcnum,len(qlist)*2,lmax+1))

for i in range(mcnum):

    print(i)

    # generate gaussian phi
    try:
        plm = pickle.load(open(D+'phi'+str(i)+'.pkl',"rb"))
    except:
        plm = curvedsky.utils.gauss1alm(clmax,ucl[3,:])
        pickle.dump((plm),open(D+'phi'+str(i)+'.pkl',"wb"),protocol=pickle.HIGHEST_PROTOCOL)


    # lensed CMB alms
    try:
        Trlm, Erlm, Brlm = pickle.load(open(D+'lcmb'+str(i)+'.pkl',"rb"))
    except:
        Talm, Ealm = curvedsky.utils.gauss2alm(clmax,ucl[0,:],ucl[1,:],ucl[2,:])
        beta = curvedsky.delens.phi2grad(npix,clmax,plm)
        Trlm, Erlm, Brlm = curvedsky.delens.remap_tp(npix,clmax,beta,np.array((Talm,Ealm,0*Ealm)))
        pickle.dump((Trlm,Erlm,Brlm),open(D+'lcmb'+str(i)+'.pkl',"wb"),protocol=pickle.HIGHEST_PROTOCOL)

    # trim cmb alms
    Trlm = Trlm[:lmax+1,:lmax+1]
    Erlm = Erlm[:lmax+1,:lmax+1]
    Brlm = Brlm[:lmax+1,:lmax+1]
    plm  = plm[:lmax+1,:lmax+1]

    # noise alms
    try:
        Tnlm, Enlm, Bnlm = pickle.load(open(D+'noise'+str(i)+'.pkl',"rb"))
    except:
        Tnlm = curvedsky.utils.gauss1alm(lmax,nl[0,:])
        Enlm = curvedsky.utils.gauss1alm(lmax,nl[1,:])
        Bnlm = curvedsky.utils.gauss1alm(lmax,nl[2,:])
        pickle.dump((Tnlm,Enlm,Bnlm),open(D+'noise'+str(i)+'.pkl',"wb"),protocol=pickle.HIGHEST_PROTOCOL)

    # obs alm
    Tolm = Trlm + Tnlm
    Eolm = Erlm + Enlm
    Bolm = Brlm + Bnlm

    # aps
    cls[i,0,:] = curvedsky.utils.alm2cl(lmax,Trlm)
    cls[i,1,:] = curvedsky.utils.alm2cl(lmax,Erlm)
    cls[i,2,:] = curvedsky.utils.alm2cl(lmax,Brlm)
    cls[i,3,:] = curvedsky.utils.alm2cl(lmax,Tnlm)
    cls[i,4,:] = curvedsky.utils.alm2cl(lmax,Enlm)
    cls[i,5,:] = curvedsky.utils.alm2cl(lmax,Bnlm)
    cls[i,6,:] = curvedsky.utils.alm2cl(lmax,plm)

    # lens norm
    Ag = {}
    for q in qlist:
        try:
            Ag[q] = np.loadtxt(D+'al_'+q+'.dat',unpack=True)[1]
        except:
            if q=='TT':  Ag[q], Ac = curvedsky.norm_lens.qtt(lmax,rlmin,rlmax,lcl[0,:rlmax+1],ocl[0,:rlmax+1])
            if q=='EE':  Ag[q], Ac = curvedsky.norm_lens.qee(lmax,rlmin,rlmax,lcl[1,:rlmax+1],ocl[1,:rlmax+1])
            if q=='EB':  Ag[q], Ac = curvedsky.norm_lens.qeb(lmax,rlmin,rlmax,lcl[1,:rlmax+1],ocl[1,:rlmax+1],ocl[2,:rlmax+1])
            if q=='MV':  
                Ag[q] = 1./(1./Ag['TT']+1./Ag['EE']+1./Ag['EB'])
                Ag[q][:3] = 0.
            np.savetxt(D+'al_'+q+'.dat',np.array((np.linspace(0,lmax,lmax+1),Ag[q])).T)

    rplm = {}
    if phirandom:
        for q in qlist:
          rplm[q] = plm + curvedsky.utils.gauss1alm(lmax,Ag[q])
    else:
        # simple diagonal c-inverse
        Fl = np.zeros((3,rlmax+1))
        for l in range(rlmin,rlmax):
            Fl[:,l] = 1./ocl[:3,l]
        # perform reconstruction
        rplm['MV'] = 0
        for qi, q in enumerate(qlist):
            try:
                rplm[q] = pickle.load(open(D+'qrec_'+q+'_'+str(i)+'.pkl',"rb"))
            except:
                # diagonal filtering (since idealistic)
                fTlm = Tolm[0:rlmax+1,0:rlmax+1]*Fl[0,:,None]
                fElm = Eolm[0:rlmax+1,0:rlmax+1]*Fl[1,:,None]
                fBlm = Bolm[0:rlmax+1,0:rlmax+1]*Fl[2,:,None]
                if q=='TT':  rplm[q], clm = curvedsky.rec_lens.qtt(lmax,rlmin,rlmax,lcl[0,:rlmax+1],fTlm,fTlm)
                if q=='EE':  rplm[q], clm = curvedsky.rec_lens.qee(lmax,rlmin,rlmax,lcl[1,:rlmax+1],fElm,fElm)
                if q=='EB':  rplm[q], clm = curvedsky.rec_lens.qeb(lmax,rlmin,rlmax,lcl[1,:rlmax+1],fElm,fBlm)
                if q!='MV':  rplm['MV'] += rplm[q]
                rplm[q] *= Ag[q][:,None]
                pickle.dump((rplm[q]),open(D+'qrec_'+q+'_'+str(i)+'.pkl',"wb"),protocol=pickle.HIGHEST_PROTOCOL)

            rls[i,qi,:] = curvedsky.utils.alm2cl(lmax,rplm[q])

    # template lensing B-mode
    Wl = {}
    for qi, q in enumerate(qlist):
        Wl[q] = np.zeros(lmax+1)
        for l in range(dlmin,dlmax+1):
            Wl[q][l] = ucl[3,l]/(ucl[3,l]+Ag[q][l])
            #Wl[q][l] = ucl[3,l]/(rls[i,qi,l])
        wplm = rplm[q]*Wl[q][:,None]
        blm  = curvedsky.delens.lensingb(lmax,dlmin,dlmax,dlmin,dlmax,Erlm[:dlmax+1,:dlmax+1],wplm[:dlmax+1,:dlmax+1])
        #blm0 = curvedsky.delens.lensingb(lmax,dlmin,dlmax,dlmin,dlmax,Erlm[:dlmax+1,:dlmax+1],plm[:dlmax+1,:dlmax+1],nside=3000)
        # aps
        dls[i,2*qi,:]   = curvedsky.utils.alm2cl(lmax,blm)
        dls[i,2*qi+1,:] = curvedsky.utils.alm2cl(lmax,Brlm-blm)


L = np.linspace(0,lmax,lmax+1)
np.savetxt(D+'cmbcls.dat',np.concatenate((L[None,:],np.mean(cls,axis=0))).T)
np.savetxt(D+'qlens.dat',np.concatenate((L[None,:],np.mean(rls,axis=0))).T)

if phirandom:
    np.savetxt(D+'resbb_random.dat',np.concatenate((L[None,:],np.mean(dls,axis=0))).T)
else:
    np.savetxt(D+'resbb.dat',np.concatenate((L[None,:],np.mean(dls,axis=0))).T)
    #np.savetxt(D+'resbb_wiener.dat',np.concatenate((L[None,:],np.mean(cls,axis=0))).T)


