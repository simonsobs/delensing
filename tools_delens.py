# Linear template delensing
import numpy as np
import healpy as hp
import pickle
import tqdm
    
# from cmblensplus
import curvedsky
import misctools

# local
import prjlib
import tools_lens
import tools_multitracer


# Define delensing filenames
class delens:

    def __init__(self,olmax=2048,elmin=20,elmax=2048,klmin=20,klmax=2048,nside=2048,klist=['TT'],kfltr='',emap='',enoise=''):

        conf = misctools.load_config('DELENSING')

        # E-mode map and noise type
        self.emap = emap
        self.enoise = enoise

        # kappa type
        self.klist = klist
        self.kfltr = kfltr

        #Newton method iteration number for obtaining anti-deflection angle in remapping
        self.nremap  = 3
        
        # minimum/maximum multipole of E and kappa in lensing template construction
        self.elmin   = conf.getint('elmin',elmin)
        self.elmax   = conf.getint('elmax',elmax)
        self.klmin   = conf.getint('klmin',klmin)
        self.klmax   = conf.getint('klmax',klmax)

        # output template
        self.olmax   = conf.getint('dklmax',olmax)

        #remapping Nside/Npix for lensing template construction / remapping
        self.nside   = nside
        self.npix    = 12*self.nside**2

        self.l  = np.linspace(0,self.olmax,self.olmax+1)


    def fname(self,qtag,mlist,etag,doreal):

        #set directory
        d = prjlib.data_directory()
        ids = prjlib.rlz_index(doreal=doreal)

        # delensing internal tag
        ltag = 'le'+str(self.elmin)+'-'+str(self.elmax)+'_lk'+str(self.klmin)+'-'+str(self.klmax)
        ttag = ltag + '_' + self.kfltr + '_' + qtag + '_' + '-'.join(mlist.keys()) + '_' + etag

        #alm of lensing template B-modes
        self.falm, self.fwlm, self.cl = {}, {}, {}
        for k in self.klist:
            self.falm[k] = [d['del']+'alm/alm_'+k+'_'+ttag+'_'+x+'.pkl' for x in ids]
            self.cl[k]   = [d['del']+'aps/rlz/cl_'+k+'_'+ttag+'_'+x+'.dat' for x in ids]

        self.gtag = '_ideal'

        # correlation coeff of templates
        self.frho = d['del'] + 'aps/rho_' + '-'.join(self.klist) + '_' + ttag



def init_template(qtag,mlist,etag,doreal,**kwargs):
    # setup parameters for lensing reconstruction (see cmblensplus/utils/quad_func.py)

    dobj = delens(**kwargs)
    delens.fname(dobj,qtag,mlist,etag,doreal)
    
    return dobj



def diag_wiener(pqf,clkk,dlmin,dlmax,kL=None,Al=None,klist=['TT','TE','EE','EB']): #kappa filter (including kappa->phi conversion)

    wlk = {}

    for k in tqdm.tqdm(klist,ncols=100,desc='load diag wiener filter'):

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

        elif k == 'comb':

            l  = np.linspace(0,dlmax,dlmax+1)
            kL = l*(l+1.)*.5
            wlk[k][dlmin:dlmax+1] = 1./kL[dlmin:dlmax+1]

        else:

            wlk[k][dlmin:dlmax+1] = 1.

    return wlk



def template_alm(rlz,klist,qf,elmin,elmax,klmin,klmax,fElm,fdlm,wlk,fgalm='',olmax=2048,glmax=2008,klist_cmb=['TT','TE','EE','EB'],**kwargs_ov):

    for k in tqdm.tqdm(klist,ncols=100,desc='template:'):
    
        for i in tqdm.tqdm(rlz,ncols=100,desc='each rlz ('+k+')',leave=False):
        
            if misctools.check_path(fdlm[k][i],**kwargs_ov): continue
            
            # load E mode
            wElm = pickle.load(open(fElm[i],"rb"))[:elmax+1,:elmax+1]
            #wElm = pickle.load(open(fElm[i].replace('base_maskv3_a5.0deg','base'),"rb"))[:elmax+1,:elmax+1]
            wElm[:elmin,:] = 0.

            # load kappa
            if k in klist_cmb:
                if qf[k].mfb is not None:
                    wplm = wlk[k][:klmax+1,None] * tools_lens.load_klms( qf[k].alm[i], klmax, fmlm=qf[k].mfb[i] )
                else:
                    wplm = wlk[k][:klmax+1,None] * tools_lens.load_klms( qf[k].alm[i], klmax )

            # This feature will be removed
            elif k == 'ALLid':
                Glm = np.load( '/project/projectdirs/sobs/delensing/multitracer_forBBgroup/coadded_tracers/combined_phi_alms_noiselessE_mvkappa_simid_'+str(i)+'.npy' )
                Glmax = 2007
                glm = 0.*wElm
                glm[20:Glmax+1,:Glmax+1] = curvedsky.utils.lm_healpy2healpix( len(Glm), Glm, Glmax )[20:,:]
                wplm = glm * wlk[k][:klmax+1,None] #* kL[:dlmax+1,None]
                wplm[:klmin,:] = 0.
                
            elif k == 'comb':
                glm = 0.*wElm
                glm[:glmax,:glmax] = pickle.load(open(fgalm[i],"rb"))
                wplm = glm * wlk[k][:klmax+1,None]
                wplm[:klmin,:] = 0.

            # construct lensing B-mode template
            dalm = curvedsky.delens.lensingb( olmax, elmin, elmax, klmin, klmax, wElm, wplm )

            # save to file
            pickle.dump((dalm),open(fdlm[k][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

            
            
def template_aps(rlz,fdlm,fBlm,fcl,W,olmax=2048,klist=['TT','TE','EE','EB'],**kwargs_ov):
    
    npix = len(W)
    nside = int(np.sqrt(npix/12.))
    
    for k in tqdm.tqdm(klist,ncols=100,desc='template aps'):
        
        for i in tqdm.tqdm(rlz,ncols=100,desc='each rlz ('+k+')',leave=False):
        
            if misctools.check_path(fcl[k][i],**kwargs_ov): continue
            
            dalm = pickle.load(open(fdlm[k][i],"rb"))[0:olmax+1,0:olmax+1]
            wdlm = curvedsky.utils.mulwin_spin(nside,olmax,olmax,2,0*dalm,dalm,W)[1]
            
            Balm = pickle.load(open(fBlm[i],"rb"))[:olmax+1,:olmax+1]
            wBlm = curvedsky.utils.mulwin_spin(nside,olmax,olmax,2,0*Balm,Balm,W)[1]
            
            clbb = curvedsky.utils.alm2cl(olmax,wBlm)
            cldd = curvedsky.utils.alm2cl(olmax,wdlm)
            clbd = curvedsky.utils.alm2cl(olmax,wdlm,wBlm)
            np.savetxt(fcl[k][i],np.array((clbb,cldd,clbd)).T)


            
def compute_coeff(rlz,fdlm,fblm,frho,W,olmax=1024,klist=['TT','TE','EE','EB']):

    npix = len(W)
    nside = int(np.sqrt(npix/12.))

    cbb = np.zeros((len(rlz),olmax+1))
    vec = np.zeros((len(rlz),len(klist),olmax+1))
    mat = np.zeros((len(rlz),len(klist),len(klist),olmax+1))
    
    #bb = 0.
    #mvec = np.zeros((len(klist),olmax+1))
    #mmat = np.zeros((len(klist),len(klist),olmax+1))
    for ii, i in enumerate(tqdm.tqdm(rlz,ncols=100,desc='compute coeff')):
        
        dalm = {}
    
        for k in klist:
            dalm[k] = pickle.load(open(fdlm[k][i],"rb"))[0:olmax+1,0:olmax+1]
            dalm[k] = curvedsky.utils.mulwin_spin(nside,olmax,olmax,2,0*dalm[k],dalm[k],W)[1]
        
        Balm = pickle.load(open(fblm[i],"rb"))[0:olmax+1,0:olmax+1]
        wBlm = curvedsky.utils.mulwin_spin(nside,olmax,olmax,2,0*Balm,Balm,W)[1]
        cbb[ii,:] = curvedsky.utils.alm2cl(olmax,wBlm)
        #bb += curvedsky.utils.alm2cl(olmax,wBlm)/len(rlz)
        
        for ki, k0 in enumerate(klist):
            vec[ii,ki,:] = curvedsky.utils.alm2cl(olmax,dalm[k0],wBlm)
            #mvec[ki,:] += curvedsky.utils.alm2cl(olmax,dalm[k0],wBlm)/len(rlz)
            for kj, k1 in enumerate(klist):
                mat[ii,ki,kj,:] = curvedsky.utils.alm2cl(olmax,dalm[k0],dalm[k1])
                #mmat[ki,kj,:] += curvedsky.utils.alm2cl(olmax,dalm[k0],dalm[k1])/len(rlz)
    
    bb, mvec, mmat = np.mean(cbb,axis=0),  np.mean(vec,axis=0),  np.mean(mat,axis=0)

    # compute correlation coefficients
    rho = np.zeros(olmax+1)
    for l in range(2,olmax):
        rho[l] = np.dot(mvec[:,l],np.dot(np.linalg.inv(mmat[:,:,l]),mvec[:,l]))

    # save to file
    L = np.linspace(0,olmax,olmax+1)
    np.savetxt(frho,np.array((L,bb,rho)).T)



def interface(run_del=[],kwargs_ov={},kwargs_cmb={},kwargs_qrec={},kwargs_mass={},kwargs_del={},klist_cmb=['TT','TE','EE','EB']):

    submap = kwargs_cmb.pop('submap')

    # //// prepare E modes //// #
    if kwargs_del['emap'] == 'id':
        pE = prjlib.analysis_init(t='id',ntype='cv')
        kwargs_del['enoise'] = 'cv'

    if kwargs_del['emap'] == 'co':
        pE = prjlib.analysis_init(t='co',submap='com',fltr='cinv',ntype=kwargs_del['enoise'].replace('_iso',''))

    # //// prepare phi //// #
    # define object
    glob = prjlib.analysis_init( submap='com', **kwargs_cmb )
    qobj = tools_lens.init_qobj( glob.stag, glob.doreal, **kwargs_qrec )
    mobj = tools_multitracer.mass_tracer( glob, qobj, **kwargs_mass )
    dobj = init_template( glob.stag+qobj.ltag, mobj.klist, pE.stag, glob.doreal, **kwargs_del )
    
    # change TT to none filter case
    #if glob.fltr == 'cinv':
    #    kwargs_cmb['fltr'] = 'none'
    #    P = prjlib.analysis_init( freq='com', **kwargs_cmb )
    #    Qobj = tools_lens.init_qobj( P.stag, P.doreal, **kwargs_qrec )
    #    qobj.f['TT'] = Qobj.f['TT']

    # pre-filtering for CMB phi
    wlk = diag_wiener( qobj.f, glob.kk, dobj.klmin, dobj.klmax, kL=glob.kL, klist=dobj.klist )

    # only kcinv for TT is used
    if dobj.kfltr == 'cinv':
        print('does not support kfltr = cinv')
    #    for k in ['TT']:
    #        wlk[k] = 1./(1e-30+p.kL[:dobj.klmax+1])
    #        qobj.f[k].alm = qobj.f[k].walm # replaced with kcinv
    #        qobj.f[k].mfb = None

    # fullsky isotropic noise
    if 'iso' in glob.ntype:
        for k in dobj.klist:  
            if k in klist_cmb:
                qobj.f[k].mfb = None
   
    # //// compute lensing template alm //// #
    if 'alm' in run_del:
        template_alm( glob.rlz, dobj.klist, qobj.f, dobj.elmin, dobj.elmax, dobj.klmin, dobj.klmax, pE.fcmb.alms['o']['E'], dobj.falm, wlk, fgalm=mobj.fcklm, olmax=dobj.olmax, **kwargs_ov )

    if 'aps' in run_del or 'rho' in run_del:
        # prepare fullsky idealistic B mode
        kwargs_cmb['t'] = 'id'
        kwargs_cmb['ntype'] = 'cv'
        pid = prjlib.analysis_init(**kwargs_cmb)
        Wsa, __ = prjlib.window('sa')
        Wla, __ = prjlib.window('la',ascale=0.)
        Wsa *= hp.pixelfunc.ud_grade(Wla,512)

    if 'aps' in run_del:
        # compute lensing template spectrum projected on SAT area #
        template_aps( glob.rlz, dobj.falm, pid.fcmb.alms['o']['B'], dobj.cl, Wsa, olmax=dobj.olmax, klist=dobj.klist, **kwargs_ov ) # ignore E-to-B leakage

    if 'rho' in run_del:
        # compute optimal combination weights
        compute_coeff( glob.rlz, dobj.falm, pid.fcmb.alms['o']['B'], dobj.frho, Wsa, olmax=dobj.olmax, klist=dobj.klist )


