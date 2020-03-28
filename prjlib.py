# Set up analysis parameters, filenames, and arrays
import numpy as np
import healpy as hp
import sys
import configparser

# from cmblensplus
import basic
import curvedsky
import constants
import cmb as CMB
import quad_func


# Define parameters
class params:

    def __init__(self,t='la',freq='145',ntype='base',lmin=2,lmax=4096,snmin=1,snmax=10,dlmin=20,dlmax=2048,rlmin=300,rlmax=4096,lTmin=500,lTmax=3000,qlist=['TT','TE','TB','EE','EB','MV'],exttag=''):

        #//// load config file ////#
        config = configparser.ConfigParser()
        if np.size(sys.argv) > 1 and '.ini' in sys.argv[1]:
            print('reading '+sys.argv[1])
            config.read(sys.argv[1])
        else:
            # default values for [QUADREC]
            config.add_section('QUADREC')
            config.set('QUADREC','oLmin','1')
            config.set('QUADREC','oLmax','3000')
            config.set('QUADREC','bn','30')
            config.set('QUADREC','binspc','p2')
            config.set('QUADREC','nside','2048')
            config.set('QUADREC','qtype','lens')
            config.set('QUADREC','rlmin',str(rlmin))
            config.set('QUADREC','rlmax',str(rlmax))
            config.set('QUADREC','n0min','1')
            config.set('QUADREC','n0max','50')
            config.set('QUADREC','rdmin','1')
            config.set('QUADREC','rdmax','100')
            config.set('QUADREC','mfmin','1')
            config.set('QUADREC','mfmax','100')

        #//// get parameters ////#
        conf = config['DEFAULT']
        self.lmin   = conf.getint('lmin',lmin)
        self.lmax   = conf.getint('lmax',lmax)
        self.olmin  = conf.getint('olmin',2)
        self.olmax  = conf.getint('olmax',lmax)
        self.bn     = conf.getint('bn',30) 
        self.binspc = conf.get('binspc','')
        self.snmin  = conf.getint('snmin',snmin)
        self.snmax  = conf.getint('snmax',snmax)
        self.doreal = conf.getboolean('doreal',False)
        self.telescope = conf.get('telescope',t)
        self.ascale = conf.getfloat('ascale',0.)
        self.lTmin  = conf.getint('lTmin',lTmin)
        self.lTmax  = conf.getint('lTmax',lTmax)

        self.freq   = conf.get('freq',freq)
        if self.telescope=='id': self.freq = '145'

        self.ntype = 'base'

        # reconstruction
        self.exttag = exttag
        self.quad  = quad_func.quad(config['QUADREC'],qlist=qlist)

        # delensing
        self.nsided  = 2048       #remapping nside for delensed map
        self.npixd   = 12*self.nsided**2
        self.dlmin   = conf.getint('dlmin',dlmin)
        self.dlmax   = conf.getint('dlmax',dlmax)
        self.nremap  = 3          #beta iteration for remapping

        #//// derived parameters ////#
        # total number of real + sim
        self.snum  = self.snmax - self.snmin + 1
        self.rlz   = np.linspace(self.snmin,self.snmax,self.snum,dtype=np.int)
        self.oL    = [self.olmin,self.olmax]
        self.dL    = [self.dlmin,self.dlmax]

        # Map resolution
        if self.telescope=='sa':
            self.nside = 512
        else:
            #self.nside = 4096
            self.nside = 2048
        self.npix = 12*self.nside**2


# Define CMB file names
class cmb:

    def __init__(self,Dir,t,nside,freq,ascale,stag,otag,ids):

        #set directory
        #d_inp = '/project/projectdirs/sobs/v4_sims/mbs/201901_gaussian_fg_lensed_cmb_realistic_noise/'
        d_alm = Dir+'cmbsims/alm/'
        d_aps = Dir+'cmbsims/aps/'
        d_map = Dir+'cmbsims/map/'
        d_msk = Dir+'mask/'

        #mask
        self.mask  = d_msk+t+'.fits'
        self.amask = d_msk+t+'_a'+str(ascale)+'deg.fits'
        if ascale==0.: self.amask = self.mask

        #cmb signal map
        if t=='id':
            self.lcdm = [d_map+'/cmb_uKCMB_la'+freq+'_nside'+str(nside)+'_'+x+'.fits' for x in ids]
        else:
            self.lcdm = [d_map+'/cmb_uKCMB_'+t+freq+'_nside'+str(nside)+'_'+x+'.fits' for x in ids]

        #cmb noise map
        self.nois = [d_map+'/noise_uKCMB_'+t+freq+'_nside'+str(nside)+'_'+x+'.fits' for x in ids]

        # cmb alm/aps
        self.alms = {}
        self.scl = {}
        self.cl = {}
        for s in ['s','n','o']:
            self.alms[s] = {}
            for m in constants.mtype:
                self.alms[s][m] = [d_alm+'/'+s+'_'+m+'_'+stag+'_'+x+'.pkl' for x in ids]
            self.scl[s] = d_aps+'aps_sim_1d_'+stag+'_'+s+'.dat'
            self.cl[s]  = [d_aps+'/rlz/cl_'+stag+'_'+s+'_'+x+'.dat' for x in ids]

        for s in ['w','W','i']:
            # filtered combined map
            # w: full wiener filtered alms
            # W: diagonal wiener-filter
            # i: ideal alms
            self.alms[s] = {}
            for m in constants.mtype:
                self.alms[s][m] = [d_alm+'/'+s+'_'+m+'_'+t+'coadd_'+x+'.pkl' for x in ids]
            self.scl[s] = d_aps+'aps_sim_1d_'+t+'coadd_'+s+'.dat'
            self.cl[s]  = [d_aps+'/rlz/cl_'+t+'coadd_'+s+'_'+x+'.dat' for x in ids]

        for s in ['x']:
            self.scl[s] = d_aps+'aps_sim_1d_'+t+'coadd_'+s+'.dat'
            self.cl[s]  = [d_aps+'/rlz/cl_'+t+'coadd_'+s+'_'+x+'.dat' for x in ids]


# * Define delensing filenames
class delensing:

    def __init__(self,Dir,dtag,otag,ids):

        #set directory
        dout = Dir+'alm/'
        dder = Dir+'derived/'

        #filename
        self.alm = [dout+'alm_'+dtag+'_'+x+'.pkl' for x in ids]
        self.wlm = [dout+'wlm_'+dtag+'_'+x+'.pkl' for x in ids]
        self.scl = dder+'scl_'+dtag+'.dat'
        self.ocl = dder+'ocl_'+dtag+'.dat'
        self.scb = dder+'scb_'+dtag+otag+'.dat'
        self.ocb = dder+'ocb_'+dtag+otag+'.dat'


# * Define class filename
class filename:

    def __init__(self,params):

        #//// root directories ////#
        Dir   = '/project/projectdirs/sobs/delensing/'
        d_cls = Dir+'official/planck_cls/'

        #//// basic tags ////#
        # map
        stag = params.telescope+params.freq+'_'+params.ntype+'_a'+str(params.ascale)+'deg'

        # output multipole range
        otag = '_oL'+str(params.olmin)+'-'+str(params.olmax)+'_b'+str(params.bn)

        #//// index ////#
        ids = [str(i).zfill(4) for i in range(201)]
        # change 1st index
        if params.doreal: ids[0] = 'real'

        #//// CAMB cls ////#
        # aps of best fit cosmology
        self.ucl = d_cls+'ffp10_scalCls.dat'
        self.lcl = d_cls+'ffp10_lensedCls.dat'

        #//// Filenames ////#
        # input phi
        self.palm = ['/global/project/projectdirs/sobs/v4_sims/mbs/cmb/input_phi/fullskyPhi_alm_0'+x+'.fits' for x in ids]

        # cmb map, window, alm and aps
        self.cmb = cmb(Dir,params.telescope,params.nside,params.freq,params.ascale,stag,otag,ids)

        # lensing reconstruction
        quad_func.quad.fname(params.quad,Dir,ids,stag+params.exttag)
        self.nkap = Dir + 'lens/phinoisemap.pkl'

        # mass tracer
        self.galm = [Dir+'multitracer_forBBgroup/combined_phi_alms_noiselessE_mvkappa_simid_'+str(i)+'.npy' for i in range(200)]

        # lensing template
        detag = 'E'+params.telescope+params.freq+'_dl'+str(params.dlmin)+'-'+str(params.dlmax)
        self.delens = {}
        for dm in ['ideal','simple','samemask','wiener']:
            self.delens[dm] = delensing(Dir+'/delensb/',dm+'_'+detag,otag,ids)


# Define arrays and functions for analysis
class recfunc:

    def __init__(self,params,filename):

        #multipole
        self.l  = np.linspace(0,params.lmax,params.lmax+1)
        self.ol = np.linspace(0,params.olmax,params.olmax+1)
        self.dl = np.linspace(0,params.dlmax,params.dlmax+1)
        self.kL = self.l*(self.l+1)*.5

        #binned multipole
        self.bp, self.bc = basic.aps.binning(params.bn,params.oL,params.binspc)

        #theoretical cl
        self.ucl = basic.aps.read_cambcls(filename.ucl,params.lmin,params.lmax,5)/constants.Tcmb**2
        self.lcl = basic.aps.read_cambcls(filename.lcl,params.lmin,params.lmax,4,bb=True)/constants.Tcmb**2

        # rename cls
        self.utt = self.ucl[0,:]
        self.uee = self.ucl[1,:]
        self.ute = self.ucl[2,:]
        self.ltt = self.lcl[0,:]
        self.lee = self.lcl[1,:]
        self.lbb = self.lcl[2,:]
        self.lte = self.lcl[3,:]

        # kappa cl
        self.pp = self.ucl[3,:]
        self.kk = self.ucl[3,:]*self.kL**2


#//////////////////////////////////////////////////
# Define some useful functions for main analysis
#//////////////////////////////////////////////////

#----------------
# initial setup
#----------------

def analysis_init(**kwargs):
    p = params(**kwargs)
    f = filename(p)
    r = recfunc(p,f)
    return p, f, r


#def filename_init(t='la',freq='145',ntype='base',lmin=2,lmax=4096,snmin=1,snmax=10,dlmin=20,dlmax=2048):
def filename_init(**kwargs):
    p = params(**kwargs)
    #p = params(t=t,freq=freq,ntype=ntype,lmin=lmin,lmax=lmax,snmin=snmin,snmax=snmax,dlmin=dlmin,dlmax=dlmax)
    f = filename(p)
    return f


#-------------------------
# SO beam, noise, window
#-------------------------

def get_beam(t,freq,lmax):
    # beam
    if t == 'sa':
        if freq == '93':   theta = 30.
        if freq == '145':  theta = 17.
        if freq == '225':  theta = 11.
        if freq == '280':  theta = 9.0

    if t == 'la' or t == 'id':
        if freq == '93':   theta = 2.2
        if freq == '145':  theta = 1.4
        if freq == '225':  theta = 1.0
        if freq == '280':  theta = 0.9

    return 1./CMB.beam(theta,lmax)


def get_polnoise_params(t,freq):

    if t == 'sa':
        if freq == '93':   sigma, lknee, alpha = 2.6, 50., -2.5
        if freq == '145':  sigma, lknee, alpha = 3.3, 50., -3.
        if freq == '225':  sigma, lknee, alpha = 6.3, 70., -3.
        if freq == '280':  sigma, lknee, alpha = 16.,100., -3.
    if t == 'la':
        lknee, alpha = 700., -1.4
        if freq == '93':   sigma = 8.
        if freq == '145':  sigma = 10.
        if freq == '225':  sigma = 22.
        if freq == '280':  sigma = 54.

    return sigma, lknee, alpha


def nlspec(t='la',freq='145',lmax=4096,ep=1e-30):
    l  = np.linspace(0,lmax,lmax+1) + ep
    bl = get_beam(t,freq,lmax)
    sigma, lknee, alpha = get_polnoise_params(t,freq)
    return l, (sigma*np.pi/10800.)**2 * ((l/lknee)**alpha+1.)/(bl+ep)**2


def nlspecs(t='la',freqs=['93','145','225','280'],ep=1e-30):
    Nl = {}
    Nl['mv'] = 0.
    for freq in freqs:
        l, Nl[freq] = nlspec(t,freq)
        Nl['mv'] += 1./(Nl[freq]+ep)
    Nl['mv'] = 1./(Nl['mv']+ep)
    return l, Nl


def nlofficial(rootdir='/project/projectdirs/sobs/delensing/official/noise/',ntype='baseline',deproj=0,cols=(1,2,3,4,5,6),dimless=False,lmax=None,lTmin=None,lTmax=None):

    # load noise at each frequency
    if deproj==-1:
        Nt = np.loadtxt(rootdir+'SO_LAT_Nell_T_'+ntype+'_fsky0p4.txt',unpack=True,usecols=cols)
        Np = np.loadtxt(rootdir+'SO_LAT_Nell_P_'+ntype+'_fsky0p4.txt',unpack=True,usecols=cols)
        # simple optimal combination
        NT = 1./np.sum(1./Nt,axis=0)
        NE = 1./np.sum(1./Np,axis=0)
        NB = 1./np.sum(1./Np,axis=0)
    if deproj>=0:
        NT = np.loadtxt(rootdir+'SO_LAT_Nell_T_'+ntype+'_fsky0p4_ILC_CMB.txt',unpack=True)[deproj+1]
        NE = np.loadtxt(rootdir+'SO_LAT_Nell_P_'+ntype+'_fsky0p4_ILC_CMB_E.txt',unpack=True)[deproj+1]
        NB = np.loadtxt(rootdir+'SO_LAT_Nell_P_'+ntype+'_fsky0p4_ILC_CMB_B.txt',unpack=True)[deproj+1]

    if dimless:
        NT /= constants.Tcmb**2
        NE /= constants.Tcmb**2
        NB /= constants.Tcmb**2

    if lmax is not None:
        NT = NT[:lmax+1]
        NE = NE[:lmax+1]
        NB = NB[:lmax+1]

    if lTmin is not None:
        NT[:lTmin] = 1e30

    if lTmax is not None:
        NT[lTmax+1:] = 1e30

    return NT, NE, NB


#---------------------------
# Window function operation
#---------------------------

def window(fmask,t='',verbose=True):

    if t=='id':
        w = 1.
    else:
        w = hp.fitsfunc.read_map(fmask,verbose=verbose)
        if t=='la':
            w = hp.pixelfunc.ud_grade(w,2048)
    w1 = np.average(w)
    w2 = np.average(w**2)
    w3 = np.average(w**3)
    w4 = np.average(w**4)
    m  = w/(w+1e-30)
    fsky = np.average(m)
    if verbose:  print(fsky,w1,w2,w3,w4)
    return w, m


def wfac(t):
    if t=='la':
        wn = np.array([0.5752538045247396,0.14110664609279655,0.046519878502273154,0.018479407525022903,0.00878600382491057])
    elif t=='sa':
        wn = np.array([0.34380340576171875,0.0816702089724941,0.039673498593158225,0.02447693801655922,0.01720146136530704])
    return wn


def multiplywindow(W,npix,nside,lmax,alm):
    T = W*curvedsky.utils.hp_alm2map(npix,lmax,lmax,alm)
    alm = curvedsky.utils.hp_map2alm(nside,lmax,lmax,T)
    return alm


def multiplywindow_spin(W,npix,nside,lmax,Ealm,Balm):
    Q, U = W*curvedsky.utils.hp_alm2map_spin(npix,lmax,lmax,2,Ealm,Balm)
    ealm, balm = curvedsky.utils.hp_map2alm_spin(nside,lmax,lmax,2,Q,U)
    return ealm, balm


#-----------------
# CMB filtering
#-----------------

def loadocl(filename,lTmin=None,lTmax=None):
    print('loading TT/EE/BB/TE from pre-computed spectrum:',filename)
    cls = np.loadtxt(filename,unpack=True,usecols=(1,2,3,4))
    if lTmin is not None:  cls[0,:lTmin] = 1e30
    if lTmax is not None:  cls[0,lTmax+1:] = 1e30
    return cls


def quad_filter(fname,lmax,lcl,sinp='w',**kwargs):

    fl = loadocl(fname[sinp],**kwargs)
    ol = loadocl(fname['o'],**kwargs)
    xl = loadocl(fname['x'],**kwargs)

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


def Wiener_Emodes(t,lmin,lmax,fcmbscl):
    '''
    Return Wiener E-mode filter
    '''

    if t=='id':
        wle = np.ones(lmax+1)
    else:
        obs = np.loadtxt(fcmbscl['o'],unpack=True,usecols=(1,2,3,4))
        sig = np.loadtxt(fcmbscl['s'],unpack=True,usecols=(1,2,3,4))
        #E-filter
        wle = np.zeros(lmax+1)
        wle[lmin:] = sig[1,lmin:]/obs[1,lmin:]
        #Noise EE
        NE = np.loadtxt(fcmbscl['n'],unpack=True)[2]
    
    return wle, NE[:lmax+1]


def getwElm(lmin,lmax,t='la',freq='145'):

    f = filename_init(t=t,freq=freq)
    return Wiener_Emodes(t,lmin,lmax,f.cmb.scl)


def Wiener_Lensing(pqf,clkk,dlmin,dlmax,kL=None,Al=None,qlist=['TT','TE','TB','EE','EB','MV']): #kappa filter (including kappa->phi conversion)

    wlk = {}

    for q in qlist:

        wlk[q] = np.zeros((dlmax+1))

        if q=='PO':
            if Al is None:
                Al0 = np.loadtxt(pqf['EE'].al,unpack=True)[1]
                Al1 = np.loadtxt(pqf['EB'].al,unpack=True)[1]
            else:
                Al0 = Al['EE']
                Al1 = Al['EB']
            Nl = 1./(1./(Al0+1e-30)+1./(Al1+1e-30))
        elif q=='TP':
            if Al is None:
                Al0 = np.loadtxt(pqf['TT'].al,unpack=True)[1]
                Al1 = np.loadtxt(pqf['TE'].al,unpack=True)[1]
                Al2 = np.loadtxt(pqf['EE'].al,unpack=True)[1]
                Al3 = np.loadtxt(pqf['EB'].al,unpack=True)[1]
            else:
                Al0 = Al['TT']
                Al1 = Al['TE']
                Al2 = Al['EE']
                Al3 = Al['EB']
            Nl = 1./(1./(Al0+1e-30)+1./(Al1+1e-30)+1./(Al2+1e-30)+1./(Al3+1e-30))
        else:
            if Al is None:
                Nl = np.loadtxt(pqf[q].al,unpack=True)[1]
            else:
                Nl = Al[q]

        for l in range(dlmin,dlmax+1):
            wlk[q][l] = clkk[l]/(clkk[l]+Nl[l])
            if kL is not None:
                wlk[q][l] /= kL[l]

    return wlk


#---------------------
# Input true phi alms
#---------------------

def load_input_plm(fpalm,lmax,verbose=False,kL=None):

    if verbose: print('load input phi alms') # true phi

    # load input phi alms
    alm = np.complex128(hp.fitsfunc.read_alm(fpalm))
    # convert order of (l,m) to healpix format
    alm = curvedsky.utils.lm_healpy2healpix(len(alm),alm,5100)[:lmax+1,:lmax+1]
    # convert to kappa alm if required
    if kL is not None:  alm *= kL[:lmax+1,None]

    return alm



