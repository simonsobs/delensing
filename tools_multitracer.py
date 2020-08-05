# Module for Multitracers
import numpy as np
import healpy as hp
import pickle
import tqdm

# from cmblensplus/wrap
import curvedsky

# from cmblensplus/utils
import misctools

# from local module
import prjlib
import tools_lens

# This file is a modified version of https://github.com/abaleato/MultitracerSims4Delensing/blob/master/multsims/MultitracerSims4Delensing.py

class mass_tracer():
    # define object which has parameters and filenames for multitracer analysis
    
    def __init__( self, glob, qobj, lmin=8, lmax=2007, add_cmb=['TT'], add_gal=np.arange(6), add_cib=True ):
        
        #lmax = 2007 #For now, set by the lmax in Byenoghee's spectra
        #lmin = 8 #For now, set by the lmin in Byenoghee's spectra

        # construct list of mass tracers to be combined
        self.klist_cmb = {}
        self.klist_gal = {}
        self.klist_cib = {}

        kid = 0
        for k in add_cmb:
            self.klist_cmb[k] = kid
            kid += 1
        
        for z in add_gal:
            self.klist_gal['g'+str(z)] = kid 
            kid += 1
        
        if add_cib: 
            self.klist_cib['cib'] = kid
        
        self.klist = { **self.klist_cmb, **self.klist_gal, **self.klist_cib }
        self.klist_ext = { **self.klist_gal, **self.klist_cib }
        #print(self.klist)
        
        self.lmin = lmin
        self.lmax = lmax

        self.nkap = len(self.klist)
        
        self.nlkk = {}
        for k, n in self.klist_cmb.items():
            self.nlkk[n] = np.loadtxt( qobj.f[k].al, unpack=True )[1][:lmax+1]

        #set directory
        d = prjlib.data_directory()
        ids = prjlib.rlz_index( doreal=glob.doreal )
 
        # cls
        self.fspec = '/global/project/projectdirs/sobs/delensing/multitracer_forBBgroup/spectra_of_tracers/'

        # kappa alm of each mass tracer
        self.fklm = {}
        for k in self.klist:
            self.fklm[k] = [ d['del'] + 'mass/' + k + '_' + str(i) + '.pkl' for i in ids ]
        
        # kappa alm of combined mass tracer
        qtag = glob.stag + qobj.ltag
        self.fcklm = [ d['del'] + 'mass/comb_' + qtag + '_' + '-'.join(self.klist.keys()) + '_' + str(i) + '.pkl' for i in ids ]


def read_phi_alms(phi_alm_file, lmax):
    phi_alm = hp.read_alm(phi_alm_file)
    return shorten_alm(phi_alm, lmax)


def shorten_alm(input_alm, lmax_new):
    lmax_old = hp.Alm.getlmax(len(input_alm))
    new_size = hp.Alm.getsize(lmax_new)
    output_alm = np.zeros(new_size, dtype=np.complex)
    
    index_in_new = np.arange(len(output_alm))
    l, m = hp.Alm.getlm(lmax_new, i=index_in_new)
    output_alm[index_in_new] = input_alm[hp.Alm.getidx(lmax_old, l, m)]
    return output_alm


def pad_cls(lmin,lmax,orig_cl):
    cl_padded = np.zeros(lmax+1)
    cl_padded[lmin:lmax+1] = orig_cl
    return cl_padded


def corrcoeff(cross, auto1, auto2):
    
    return cross/np.sqrt(auto1*auto2)


def calculate_sim_weights( cl, lmin, num_of_kcmb ):
    '''
    Calculate the weights A_l^{ij} and the auxiliary spectra C_l^{ij}={C_l^{uu},C_l^{ee},...} from which the to draw the alm coefficients a_p={u_{lm},e_{lm},...}
    The simulated alm has the form, alm = sum_{p=0}^i A^{ip} a^p, where a^p is the auxiliary alm. To abvoid completely degenerate case for CMB estimators,
    we set A^{ip} = 0 for p within p > 0 and p < num of kcmb.
    '''
    num_of_tracers = len(cl[:,0,0]) 
    num_of_multipoles = len(cl[0,0,:])
    aux_cl = np.zeros( (num_of_tracers, num_of_multipoles) ) #Auxiliary spectra
    A = np.zeros( (num_of_tracers,num_of_tracers,num_of_multipoles) ) #Weights for the alms
    #aux_cl = np.zeros((num_of_tracers, num_of_multipoles), dtype='complex128') #Auxiliary spectra
    #A = np.zeros((num_of_tracers,num_of_tracers,num_of_multipoles), dtype='complex128') #Weights for the alms

    for j in range(num_of_tracers):

        for i in range(num_of_tracers):
        
            if j>i:
            
                pass
            
            else:

                aux_cl[j,:] = np.nan_to_num(cl[j,j,:])

                if (i > 0 and i < num_of_kcmb) or (j > 0 and j < num_of_kcmb) : continue

                for p in range(j):
                    aux_cl[j] -= np.nan_to_num(A[j,p,:]**2 * aux_cl[p,:])

                A[i,j,lmin:] = np.nan_to_num((1./aux_cl[j,lmin:])*cl[i,j,lmin:])

                for p in range(j):
                    A[i,j,lmin:] -= np.nan_to_num((1./aux_cl[j,lmin:])*A[j,p,lmin:]*A[i,p,lmin:]*aux_cl[p,lmin:])
    
    return aux_cl, A


def draw_gaussian_a_p(input_kappa_alm, aux_cl, num_of_kcmb):
    '''
    Draw a_p alms from distributions with the right auxiliary spectra.
    '''
    num_of_tracers = len(aux_cl[:,0])
    a_alms = np.zeros((num_of_tracers, len(input_kappa_alm)), dtype='complex128') #Unweighted alm components

    a_alms[0:num_of_kcmb,:] = input_kappa_alm
    for j in range(num_of_kcmb, num_of_tracers):
        a_alms[j,:] = hp.synalm(aux_cl[j,:], lmax=len(aux_cl[0,:])-1)

    return a_alms


def generate_individual_gaussian_tracers(a_alms, A, nlkk, num_of_kcmb):
    '''
    Put all the weights and alm components together to give appropriately correlated tracers
    '''
    num_of_tracers = len(a_alms[:,0])
    tracer_alms = np.zeros((num_of_tracers, len(a_alms[0,:])), dtype='complex128') #Appropriately correlated final tracers

    for i in range(num_of_kcmb):
        tracer_alms[i,:] = a_alms[i,:] + hp.synalm(nlkk[i], lmax=len(A[i,i,:])-1)
    
    for i in range(num_of_kcmb,num_of_tracers):
        for j in range(i+1):
            tracer_alms[i,:] += hp.almxfl(a_alms[j,:], A[i,j,:])

    return tracer_alms



def calculate_multitracer_weights(spectra_matrix, clkk, lmin):
    '''
    Calculate the weights in the way described in Blake and Marcel's paper
    '''
    num_of_tracers = len(spectra_matrix[:,0,0])
    num_of_multipoles = len(spectra_matrix[0,0,:])
    tracer_corr_matrix = np.ones((num_of_tracers, num_of_tracers, num_of_multipoles))
    inv_tracer_corr_matrix = np.zeros(tracer_corr_matrix.shape)
    tracer_corr_w_phi = np.zeros((num_of_tracers,num_of_multipoles))
    c_array = np.zeros((num_of_tracers, num_of_multipoles))

    for i in range(num_of_tracers):
        for j in range(num_of_tracers):
            if j>i:
                tracer_corr_matrix[i,j,:] = tracer_corr_matrix[j,i,:] = corrcoeff(spectra_matrix[i,j,:], spectra_matrix[i,i,:], spectra_matrix[j,j,:])
            else:
                pass

    for k in range(num_of_multipoles):
        try:
            inv_tracer_corr_matrix[:,:,k] = np.linalg.inv(tracer_corr_matrix[:,:,k])
        except:
            pass

    for t in range(num_of_tracers):
        if t is 0:
            # Should probably make this neater...
            tracer_corr_w_phi[t,:] = corrcoeff(clkk, spectra_matrix[t,t,:], clkk)
        else:
            tracer_corr_w_phi[t,:] = corrcoeff(spectra_matrix[0,t,:], spectra_matrix[t,t,:], clkk)

    for index in range(num_of_tracers):
        for l in range(lmin,num_of_multipoles):
            c_array[index,l] = np.dot(tracer_corr_w_phi[:,l],inv_tracer_corr_matrix[index,:,l])*np.sqrt(clkk/spectra_matrix[index,index,:])[l]

    return c_array



def coadd_kappa_alms(tracer_alms, weights):
    
    combined_kappa_alms = 0.*tracer_alms[0,:,:]
    
    for index, individual_alms in enumerate(tracer_alms):
        combined_kappa_alms +=  weights[index,:,None] * individual_alms
    
    return combined_kappa_alms
        


def get_spectra_matrix( mobj ):
    
    lmin = mobj.lmin
    lmax = mobj.lmax

    #The spectra below have been generated by Byeonghee Yu
    # LSST auto
    clgg = np.loadtxt(mobj.fspec+"clgg_LSSTgold_6tomobins.dat") # dimension = (6,2000) = (gg bin, ell bin)
    # e.g. clgg_bin1 = clgg_matrix[0,:]; clgg_bin2 = clgg_matrix[1,:]; clgg_bin3 = clgg_matrix[2,:]; clgg_bin4 = clgg_matrix[3,:]; clgg_bin5 = clgg_matrix[4,:]; clgg_bin6 = clgg_matrix[5,:]
    # LSST x kappa
    clkg = np.loadtxt(mobj.fspec+"clkg_LSSTgold_6tomobins.dat") # dimension = (6,2000) = (gg bin, ell bin)
    # LSSTxCIB
    clgI = np.loadtxt(mobj.fspec+"cl_CIBxLSSTgold_6tomobins.dat") # dimension = (6,2000) = (gg bin, ell bin)
    # Clkk
    clkk = np.loadtxt(mobj.fspec+"cl_kk.dat") # a vector of length 2000, ellmin = 8, ellmax = 2007
    # CIB auto
    clII = np.loadtxt(mobj.fspec+"cl_CIBauto.dat") # a vector of length 2000, ellmin = 8, ellmax = 2007
    # CIB x kappa
    clkI = np.loadtxt(mobj.fspec+"cl_CIBxkappa.dat") # a vector of length 2000, ellmin = 8, ellmax = 2007

    # used for generating sim
    cl_matrix   = np.zeros( ( mobj.nkap, mobj.nkap, lmax+1) ) #Theory auto and cross spectra
    #cl_matrix   = np.zeros( ( mobj.nkap, mobj.nkap, lmax+1), dtype='complex128' ) #Theory auto and cross spectra

    # //// auto spectra //// #
    for n in mobj.klist_cmb.values():
        cl_matrix[n,n,:] = pad_cls(lmin,lmax,clkk)
    
    for k, n in mobj.klist_gal.items():
        z = int(k[1])
        cl_matrix[n,n,:] = pad_cls(lmin,lmax,clgg[z,:])
    
    for n in mobj.klist_cib.values():
        cl_matrix[n,n,:] = pad_cls(lmin,lmax,clII)

    # //// cross spectra //// #
    for n0 in mobj.klist_cmb.values():
        for n1 in mobj.klist_cmb.values():
            if n1 > n0: 
                continue
            cl_matrix[n0,n1,:] = cl_matrix[n1,n0,:] = pad_cls(lmin,lmax,clkk)
    
    for n0 in mobj.klist_cmb.values():
        for j, n1 in mobj.klist_gal.items():
            z = int(j[1])
            cl_matrix[n0,n1,:] = cl_matrix[n1,n0,:] = pad_cls(lmin,lmax,clkg[z,:])

    for n0 in mobj.klist_cmb.values():
        for n1 in mobj.klist_cib.values():
            cl_matrix[n0,n1,:] = cl_matrix[n1,n0,:] = pad_cls(lmin,lmax,clkI)

    for n0 in mobj.klist_cib.values():
        for j, n1 in mobj.klist_gal.items():
            z = int(j[1])
            cl_matrix[n0,n1,:] = cl_matrix[n1,n0,:] = pad_cls(lmin,lmax,clgI[z,:])


    # used for weights for coadding
    clnl_matrix = cl_matrix.copy()
    for n in mobj.klist_cmb.values():
        clnl_matrix[n,n,:] += mobj.nlkk[n]

    return cl_matrix, clnl_matrix


def generate_tracer_alms(simid, cl_matrix, iklm, num_of_kcmb, nlkk = 0., lmin=8, lmax=2007):

    # Calculate the weights and auxiliary spectra needed to generate Gaussian sims of individual tracers
    aux_cl, A = calculate_sim_weights( cl_matrix, lmin, num_of_kcmb )

    # Draw harmonic coefficients from Gaussian distributions with the calculated auxiliary spectra
    a_alms = draw_gaussian_a_p( iklm, aux_cl, num_of_kcmb )

    # Combine weights and coefficients to generate sims of individual tracers
    tracer_alms = generate_individual_gaussian_tracers( a_alms, A, nlkk, num_of_kcmb )

    return tracer_alms


def interface( run=['gen_alm','comb'], kwargs_ov={}, kwargs_cmb={}, kwargs_qrec={}, kwargs_mass={} ):

    # load parameters and filenames
    kwargs_cmb['t']    = 'la'
    kwargs_cmb['submap'] = 'com'
    glob = prjlib.analysis_init( **kwargs_cmb )
    qobj = tools_lens.init_qobj( glob.stag, glob.doreal, **kwargs_qrec )
    mobj = mass_tracer( glob, qobj, **kwargs_mass )
    
    # load LAT window
    if 'iso' in glob.ntype:
        win = None
    else:
        win, __ = prjlib.window('la',ascale=5.)
        nside = hp.pixelfunc.get_nside(win)

    # load cl-matrix and covariance of alms
    cl_matrix, clnl_matrix = get_spectra_matrix( mobj )
    
    # generate random gaussian alms
    if 'gen_alm' in run:
            
        # loop over realizations
        for i in tqdm.tqdm(glob.rlz,ncols=100,desc='generating multitracer klms'):
        
            # load input phi alm and then convert it to kappa alm
            iplm = read_phi_alms( glob.fpalm[i], mobj.lmax )
            iklm = hp.almxfl( iplm, glob.kL[:mobj.lmax+1] )

            # generate tracer alms
            tracer_alms = generate_tracer_alms( i, cl_matrix, iklm, len(mobj.klist_cmb), nlkk=mobj.nlkk )

            # save to files
            for k, n in mobj.klist_ext.items():
                
                if misctools.check_path(mobj.fklm[k][i],**kwargs_ov): continue
            
                # re-ordering l,m to match healpix
                alms = curvedsky.utils.lm_healpy2healpix( len(tracer_alms[n,:]), tracer_alms[n,:], mobj.lmax )
                
                pickle.dump( (alms), open(mobj.fklm[k][i],"wb"), protocol=pickle.HIGHEST_PROTOCOL )
            
            
    # Co-add the individual tracers using the weights we just calculated
    if 'comb' in run:
        
        # Calculate the optimal weights to form a multitracer map for delensing
        c_array = calculate_multitracer_weights( clnl_matrix, cl_matrix[0,0,:], mobj.lmin )
        
        # loop over realizations
        for i in tqdm.tqdm(glob.rlz,ncols=100,desc='coadding multitracer'):
            
            if misctools.check_path(mobj.fcklm[i],**kwargs_ov): continue
                
            alms = np.zeros( ( mobj.nkap, mobj.lmax+1, mobj.lmax+1 ), dtype=np.complex )
            
            for k, n in mobj.klist_cmb.items():
                if 'iso' in glob.ntype:
                    alms[n,:,:] = tools_lens.load_klms( qobj.f[k].alm[i], mobj.lmax )
                else:
                    alms[n,:,:] = tools_lens.load_klms( qobj.f[k].alm[i], mobj.lmax, fmlm = qobj.f[k].mfb[i] )

            for k, n in mobj.klist_ext.items():
                alms[n,:,:] = pickle.load( open(mobj.fklm[k][i],"rb") )
                if win is not None:
                    alms[n,:,:] = curvedsky.utils.mulwin( nside, mobj.lmax, mobj.lmax, alms[n,:,:], win**2 )
                
            # coadd
            cklms = coadd_kappa_alms( alms, c_array )
            
            # save to a file
            pickle.dump( (cklms), open(mobj.fcklm[i],"wb"), protocol=pickle.HIGHEST_PROTOCOL )

    

