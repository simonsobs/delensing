#///////////////////////////////////////////////////////////////////////////////////////////////////#
# This file is intended to provide parametes, functions, etc, affecting the delensing code globally #
# Set up analysis parameters, filenames, arrays, functions                                          #
#///////////////////////////////////////////////////////////////////////////////////////////////////#

import numpy as np
import healpy as hp
import sys
import configparser

# from cmblensplus/wrap/
import basic
import curvedsky

# from cmblensplus/utils/
import constant as constants # Name changed in latest cmblensplus version
import cmb as CMB
import misctools


#////////// Define Fixed Values //////////#

def data_directory():
    
    direct = {}

    root = '/project/projectdirs/sobs/delensing_new/' # E.H.: changed directory name to delensing_new
    direct['root'] = root
    direct['cls']  = root + 'official/planck_cls/'
    direct['win']  = root + 'mask/'
    direct['hit']  = root + 'hitmap/'
    direct['cmb']  = root + 'cmbsims/'
    direct['del']  = root + 'delensb/'
    direct['pub']  = root + 'official/'

    return direct


def mapres(telescope):
    # Map resolution (Nisde and Npix for Healpix)
    # A lower resolution is applied to SA
    # Otherwise, resolution is fixed to Nside = 2048
    if telescope.lower() == 'sa':  
        nside = 512
    else:  
        nside = 2048
    # Total number of Healpix pixels
    npix = 12*nside**2

    return nside, npix


def rlz_index(doreal=False):
    
    # Index for realizations e.g. 0001
    ids = [str(i).zfill(4) for i in range(201)] # E.H.: might have to change this to do less (or more) RLZs

    # change 1st index if real data is used
    if doreal: 
        ids[0] = 'real'

    return ids



# Define CMB file names
class cmb:

    def __init__(self,t,nside,freq,ntype,stag,doreal):

        #set directory
        d = data_directory()
        d_alm = d['cmb'] + 'alm/'
        d_aps = d['cmb'] + 'aps/'
        d_map = d['cmb'] + 'map/'

        ids = rlz_index(doreal=doreal)

        #cmb signal map
        if t=='id': # use LAT signal sim
            self.lcdm = [d_map+'cmb_uKCMB_la145_nside'+str(nside)+'_'+x+'.fits' for x in ids]
        else:
            self.lcdm = [d_map+'cmb_uKCMB_'+t+freq+'_nside'+str(nside)+'_'+x+'.fits' for x in ids]

        #cmb noise map
        self.nois = [d_map+'noise_uKCMB_'+t+freq+'_'+ntype+'_nside'+str(nside)+'_'+x+'.fits' for x in ids]

        #cmb alm/aps
        self.alms = {}
        self.scl  = {} # self.scl to be removed
        self.cl   = {}
        for s in ['s','n','o']:
            self.alms[s] = {}
            for m in constants.mtype:
                if s=='s': # remove noise type for signal
                    Stag = stag.replace('_'+ntype,'')
                else:
                    Stag = stag
                self.alms[s][m] = [d_alm+'/'+s+'_'+m+'_'+Stag+'_'+x+'.pkl' for x in ids]
            self.scl[s] = d_aps+Stag+'_'+s+'.dat'
            self.cl[s]  = [d_aps+'rlz/cl_'+Stag+'_'+s+'_'+x+'.dat' for x in ids]

        #for cross spectrum with input alm
        for s in ['x']:
            self.scl[s] = d_aps+stag+'_'+s+'.dat'
            self.cl[s]  = [d_aps+'rlz/cl_'+stag+'_'+s+'_'+x+'.dat' for x in ids]


# Define parameters, filename and array
# E.H.: snmin and snmax correspond to RLZ indices
class analysis:

    def __init__(self,t='la',freq='',ntype='base_roll50',fltr='none',lmin=2,snmin=1,snmax=10,lTmin=500,lTmax=3000,ascale=5.):

        #//// load config file ////#
        #config = configparser.ConfigParser()
        #if np.size(sys.argv) > 1 and '.ini' in sys.argv[1]:
        #    print('reading '+sys.argv[1])
        #    config.read(sys.argv[1])
        #else:
        #    config.add_section('DEFAULT')

        #//// get parameters ////#
        conf = misctools.load_config('CMB')

        # specify telescope
        # la --- LAT (default)
        # sa --- SAT
        # co --- SAT + LAT
        # id --- fullsky, isotropic noise
        self.telescope = conf.get('telescope',t)

        # minimum/maximum of realization index to be analyzed
        # the 1st index (0000) is used for real (or mock) data
        self.snmin  = conf.getint('snmin',snmin)
        self.snmax  = conf.getint('snmax',snmax)

        # use real data or not for index = 0000
        self.doreal = conf.getboolean('doreal',False)
        
        # total number of realizations and array of realization index
        self.snum  = self.snmax - self.snmin + 1
        self.rlz   = np.linspace(self.snmin,self.snmax,self.snum,dtype=np.int)

        # CMB frequency
        self.freq = conf.get('freq',freq)
        if self.telescope=='id':  self.freq = '145'
  
        # CMB alms filtering
        self.fltr = conf.get('fltr',fltr)
            
        # apodization scale
        self.ascale = conf.getfloat('ascale',ascale)
        if self.telescope=='id':  self.ascale = 0.

        # CMB map noise type 
        # base --- SO baseline (default)
        # goal --- SO goal
        self.ntype = conf.get('ntype',ntype)
        # Note that you can also specify roll-off effect on large scales to mimic actual map-making
        # e.g. ntype = base_roll200  ---  basline noise + roll-off effect below ell<200
        
        # set roll-off multipole
        # assuming "base_roll200", etc
        if 'roll' in ntype:
            self.roll = int(ntype[ntype.find('roll')+4:])
        else:
            self.roll = 0

        # minimum/maximum multipoles of CMB alms
        self.lmin   = conf.getint('lmin',lmin)
        # maximum multipole of CMB maps are fixed for each telescope
        if t in ['sa','co']:  
            self.lmax = 2048 
        else:
            self.lmax = 4096

        # minimum/maximum multipoles of CMB temperature alms for lensing reconstruction
        self.lTmin  = conf.getint('lTmin',lTmin)
        self.lTmax  = conf.getint('lTmax',lTmax)

        #//// derived parameters ////#
        self.nside, self.npix = mapres(self.telescope)


    def filename(self):  #construct filename from parameters

        # set directory
        d = data_directory()

        #//// tags for filename ////#
        # type of window
        if self.telescope == 'id':
            wftag = ''
        elif self.telescope == 'co':
            wftag = '_'+self.fltr
        else:
            wftag = '_mv3' + '_a'+str(self.ascale)+'deg_' + self.fltr
        
        #ftag = ''
        #if self.fltr != '': ftag = '_'+self.fltr
        
        # specify CMB map
        self.stag = self.telescope + self.freq + '_' + self.ntype + wftag
        #self.stag = self.telescope + self.freq + ftag + '_' + self.ntype + wtag 

        # index for realizations e.g. 0001
        ids = rlz_index(doreal=self.doreal)

        #//// CAMB cls ////#
        # aps of best fit cosmology (currently PLANCK FFP10)
        self.fucl = d['cls']+'ffp10_scalCls.dat'
        self.flcl = d['cls']+'ffp10_lensedCls.dat'

        #//// Filenames ////#
        # input phi
        self.fpalm = ['/global/project/projectdirs/sobs/v4_sims/mbs/cmb/input_phi/fullskyPhi_alm_0'+x+'.fits' for x in ids]

        # filename of survey window
        self.fmask = window_name(self.telescope,ascale=self.ascale)

        # cmb map, alm and aps
        self.fcmb = cmb(self.telescope,self.nside,self.freq,self.ntype,self.stag,self.doreal)


    def array(self):  #construct array from parameters

        #multipole
        self.l  = np.linspace(0,self.lmax,self.lmax+1)

        #conversion factor from phi to kappa
        self.kL = self.l*(self.l+1)*.5

        #loading theoretical cl
        self.ucl = CMB.read_camb_cls(self.fucl,ftype='scal',output='array')[:,:self.lmax+1]
        self.lcl = CMB.read_camb_cls(self.flcl,ftype='lens',output='array')[:,:self.lmax+1]

        #rename cls
        self.uTT = self.ucl[0,:]
        self.uEE = self.ucl[1,:]
        self.uTE = self.ucl[2,:]
        self.lTT = self.lcl[0,:]
        self.lEE = self.lcl[1,:]
        self.lBB = self.lcl[2,:]
        self.lTE = self.lcl[3,:]

        #kappa cl
        self.pp = self.ucl[3,:]
        self.kk = self.ucl[3,:]*self.kL**2


#//////////////////////////////////////////////////
# Define some useful functions for main analysis
#//////////////////////////////////////////////////

#----------------
# initial setup
#----------------

def analysis_init(**kwargs):
    # setup parameters, filenames, and arrays
    p = analysis(**kwargs)
    analysis.filename(p)
    analysis.array(p)
    return p



def filename_freqs(freqs,**kwargs):
    # setup cmb filenames for frequencies
    ffreq = {}
    for freq in freqs:
        fnu = analysis_init(freq=freq,**kwargs)
        ffreq[freq] = fnu.fcmb
    return ffreq


#-------------------------
# SO beam, noise, window
#-------------------------

def get_beam(t,freq,lmax): # Return Gaussian beam function

    if t == 'sa': #SAT beam FWHM in arcmin
        if freq == '93':   theta = 30.
        if freq == '145':  theta = 17.
        if freq == '225':  theta = 11.
        if freq == '280':  theta = 9.0

    if t == 'la': #LAT beam FWHM in arcmin
        if freq == '93':   theta = 2.2
        if freq == '145':  theta = 1.4
        if freq == '225':  theta = 1.0
        if freq == '280':  theta = 0.9

    if t == 'id': #use LAT signal sims at 145GHz
        theta = 1.4

    # compute 1D Gaussian beam function from cmblensplus/utils/cmb.py
    return 1./CMB.beam(theta,lmax)


def get_polnoise_params(t,freq): # Return parameters for SO polarization noise

    # sigma = \sigma_P in muK-arcmin
    # lknee = knee multipole of 1/f noise
    # alpha = power of 1/f noise
    
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
    # Compute SO noise spectrum analytically
    # ep: a very small value to avoid division by 0

    # multipole
    l  = np.linspace(0,lmax,lmax+1) + ep

    # 1D Gaussian beam
    bl = get_beam(t,freq,lmax)

    # noise parameters
    sigma, lknee, alpha = get_polnoise_params(t,freq)

    # multipole and noise spectrum
    return l, (sigma*np.pi/10800.)**2 * ((l/lknee)**alpha+1.)/(bl+ep)**2


def nlspecs(t='la',freqs=['93','145','225','280'],ep=1e-30):
    # Compute SO noise spectrum analytically (combined noise from frequencies)

    # initialize
    Nl = {}
    Nl['mv'] = 0.

    # noise spectrum for each frequency
    for freq in freqs:
        l, Nl[freq] = nlspec(t,freq)
        Nl['mv'] += 1./(Nl[freq]+ep)
    
    # a simple combined noise spectrum
    Nl['mv'] = 1./(Nl['mv']+ep)
    
    return l, Nl


def nlofficial(ntype='baseline',deproj=0,cols=(1,2,3,4,5,6),dimless=False,lmax=None,lTmin=None,lTmax=None):
    # Load official SO noise spectrum

    # set directory
    d = data_directory()
    rootdir = d['pub'] + 'noise/'

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

    if dimless: # remove uK^2
        NT /= constants.Tcmb**2
        NE /= constants.Tcmb**2
        NB /= constants.Tcmb**2

    if lmax is not None: # restrict output lmax
        NT = NT[:lmax+1]
        NE = NE[:lmax+1]
        NB = NB[:lmax+1]

    if lTmin is not None: # set very large value for l<lTmin for temperature noise
        NT[:lTmin] = 1e30

    if lTmax is not None: # set very large value for lTmax<l for temperature noise
        NT[lTmax+1:] = 1e30

    return NT, NE, NB


#---------------------------
# Window function operation
#---------------------------

def window_name(t,ascale=0):

    d = data_directory()
    
    #if t == 'la':  fmask_org = d['win'] + 'simonsobs_noise_mask_x_mask_V3_bool.fits'
    #if t == 'sa':  fmask_org = d['win'] + 'mask_04000.fits'
    #if t == 'sa':  fmask_org = d['win'] + 'mask_apodized.fits'

    fmask_rev  = d['win']+t+'_binary.fits'
    #famask = d['win']+t+'_n'+str(nside).zfill(4)+'_a'+str(ascale)+'deg.fits'

    if ascale==0: fmask = fmask_rev
    if ascale!=0: fmask = fmask_rev.replace('binary','a'+str(ascale)+'deg')

    return fmask


def window(t,nside=None,ascale=0.,ep=1e-30):

    # load window
    if t=='id':
        w = 1.
    else:
        fmask = window_name(t,ascale)
        w = hp.fitsfunc.read_map(fmask,verbose=False)

        if nside is not None:  
            w = hp.pixelfunc.ud_grade(w,nside)
    
    # normalization correction due to window
    wn = np.zeros(5)
    for n in range(1,5):
        wn[n] = np.average(w**n)

    # binary mask
    m  = w/(w+ep)
    wn[0] = np.average(m)

    return w, wn


def wfac(t,binary=False):
    
    if t=='la':
        #wn = np.array([0.5752538045247396,0.14110664609279655,0.046519878502273154,0.018479407525022903,0.00878600382491057])
        wn = np.array([0.2910047173500061,0.25404106812564026,0.24660663194819224,0.24293087378051578,0.24061642729389157])
    elif t=='sa':
        wn = np.array([0.34380340576171875,0.0816702089724941,0.039673498593158225,0.02447693801655922,0.01720146136530704])
    elif t=='co':
        wn = np.array([0.57702001,0.11138843,0.03370565,0.01320226,0.00612444])
    else:
        wn = np.ones(5)

    if binary:
        wn[:] = wn[0]
    
    return wn


#-----------------
# Hit Count Map
#-----------------

def hitmap_filename(telescope,nside):

    d = data_directory()
    f = d['hit'] + telescope.lower() + '_n'+str(nside).zfill(4)+'.fits'
    return f


def hitmap(telescope,nside,verbose=True):

    # load window
    f = hitmap_filename(telescope,nside)
    w = hp.fitsfunc.read_map(f,verbose=verbose)

    return w


def loadocl(filename,lTmin=None,lTmax=None):

    print('loading TT/EE/BB/TE from pre-computed spectrum:',filename)
    
    cls = np.loadtxt(filename,unpack=True,usecols=(1,2,3,4))
    
    if lTmin is not None:  cls[0,:lTmin] = 1e30
    if lTmax is not None:  cls[0,lTmax+1:] = 1e30
    
    return cls


#---------------------
# Input true phi alms
#---------------------

def load_input_plm(fpalm,lmax,verbose=False,ktype=''):

    if verbose: print('load input phi alms') # true phi

    # load input phi alms
    alm = np.complex128(hp.fitsfunc.read_alm(fpalm))
    # convert order of (l,m) to healpix format
    alm = curvedsky.utils.lm_healpy2healpix(alm,5100,len(alm))[:lmax+1,:lmax+1]
    # convert to kappa alm if required
    if ktype == 'k':
        L  = np.linspace(0,lmax,lmax+1)
        kL = L*(L+1)*.5
        alm *= kL[:,None]

    return alm


#---------------------
# Plot map
#---------------------

def view_map_from_alm(alm,nside,lmax,min=-.1,max=.1):
    Map = curvedsky.utils.hp_alm2map(nside,lmax,lmax,alm[:lmax+1,:lmax+1])
    hp.mollview(Map,min=min,max=max)


