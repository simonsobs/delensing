import numpy as np
import healpy as hp

def read_phi_alms(idx, lmax, path=''):
    phi_alm_file = path + 'fullskyPhi_alm_'+ ('%05d' % idx) +'.fits'
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

def pad_cls(orig_cl):
    cl_padded = np.zeros(ells.shape)
    cl_padded[lmin:lmax+1] = orig_cl
    return cl_padded

def corrcoeff(cross, auto1, auto2):
    return cross/np.sqrt(auto1*auto2)

def calculate_sim_weights(cl):
    '''
    Calculate the weights A_l^{ij} and the auxiliary spectra C_l^{ij}={C_l^{uu},C_l^{ee},...} from which the to draw the alm coefficients a_p={u_{lm},e_{lm},...}
    '''
    num_of_tracers = len(cl[:,0,0])
    num_of_multipoles = len(cl[0,0,:])
    aux_cl = np.zeros((num_of_tracers, num_of_multipoles), dtype='complex128') #Auxiliary spectra
    A = np.zeros((num_of_tracers,num_of_tracers,num_of_multipoles), dtype='complex128') #Weights for the alms

    for j in range(num_of_tracers):
        for i in range(num_of_tracers):
            if j>i:
                pass
            else:
                aux_cl[j,:] = np.nan_to_num(cl[j,j,:])
                for p in range(j):
                    aux_cl[j] -= np.nan_to_num(A[j,p,:]**2 * aux_cl[p,:])

                A[i,j,:] = np.nan_to_num((1./aux_cl[j,:])*cl[i,j,:])

                for p in range(j):
                    A[i,j,:] -= np.nan_to_num((1./aux_cl[j,:])*A[j,p,:]*A[i,p,:]*aux_cl[p,:])
    return aux_cl, A

def draw_gaussian_a_p(input_kappa_alm, aux_cl):
    '''
    Draw a_p alms from distributions with the right auxiliary spectra.
    '''
    num_of_tracers = len(aux_cl[:,0])
    a_alms = np.zeros((num_of_tracers, len(input_kappa_alm)), dtype='complex128') #Unweighted alm components

    a_alms[0,:] = input_kappa_alm
    for j in range(1, num_of_tracers):
        a_alms[j,:] = hp.synalm(aux_cl[j,:], lmax=len(aux_cl[0,:])-1)

    return a_alms

def generate_individual_gaussian_tracers(a_alms, A, nlkk):
    '''
    Put all the weights and alm components together to give appropriately correlated tracers
    '''
    num_of_tracers = len(a_alms[:,0])
    tracer_alms = np.zeros((num_of_tracers, len(a_alms[0,:])), dtype='complex128') #Appropriately correlated final tracers

    tracer_alms[0,:] = a_alms[0,:] + hp.synalm(nlkk, lmax=len(A[0,0,:])-1)
    for i in range(1,num_of_tracers):
        for j in range(i+1):
            tracer_alms[i,:] += hp.almxfl(a_alms[j,:], A[i,j,:])

    return tracer_alms

def calculate_multitracer_weights(spectra_matrix, clkk):
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

    for k in range(len(pad_cls(clkk))):
        try:
            inv_tracer_corr_matrix[:,:,k] = np.linalg.inv(tracer_corr_matrix[:,:,k])
        except:
            pass

    for t in range(num_of_tracers):
        if t is 0:
            # Should probably make this neater...
            tracer_corr_w_phi[t,:] = corrcoeff(pad_cls(clkk), spectra_matrix[t,t,:], pad_cls(clkk))
        else:
            tracer_corr_w_phi[t,:] = corrcoeff(spectra_matrix[0,t,:], spectra_matrix[t,t,:], pad_cls(clkk))

    for index in range(num_of_tracers):
        for l in np.arange(num_of_multipoles):
            c_array[index,l] = np.dot(tracer_corr_w_phi[:,l],inv_tracer_corr_matrix[index,:,l])*np.sqrt(pad_cls(clkk)/spectra_matrix[index,index,:])[l]

    return c_array

def get_coadded_kappa_tracer(tracer_alms, weights):
    '''
    Form a linear combination of the individual kappas using the weights
    '''
    combined_kappa_alms = np.zeros(len(tracer_alms[0,:]), dtype=np.complex)
    for index, individual_alms in enumerate(tracer_alms):
        combined_kappa_alms += np.nan_to_num(hp.almxfl(individual_alms, weights[index,:]))
    return combined_kappa_alms


def generate_multitracer_phi(simid, spectra_path = '/global/u1/a/ab2368/correlated_tracers/', signal_path = '/global/cscratch1/sd/engelen/simsS1516_v0.4/data/', output_path = '.', save=False):
    signal_path = signal_path #Location of input phi alms
    spectra_path = spectra_path
    output_path = output_path

    simid = simid #int(sys.argv[1])
    lmax = 2007 #For now, set by the lmax in Byenoghee's spectra
    lmin = 8 #For now, set by the lmin in Byenoghee's spectra
    ells = np.arange(lmax+1)
    nside_out = 1024

    #Load the input phi alms, shorten to appropriate lmax and convert to kappa
    input_phi_alm = read_phi_alms(simid, lmax, signal_path)
    input_kappa_alm = hp.almxfl(input_phi_alm, (1.0/2)*ells**2)

    #The spectra below have been generated by Byeonghee Yu
    # LSST auto
    clgg_matrix = np.loadtxt(spectra_path+"clgg_LSSTgold_6tomobins.dat") # dimension = (6,2000) = (gg bin, ell bin)
    # e.g. clgg_bin1 = clgg_matrix[0,:]; clgg_bin2 = clgg_matrix[1,:]; clgg_bin3 = clgg_matrix[2,:]; clgg_bin4 = clgg_matrix[3,:]; clgg_bin5 = clgg_matrix[4,:]; clgg_bin6 = clgg_matrix[5,:]
    # LSST x kappa
    clkg_matrix = np.loadtxt(spectra_path+"clkg_LSSTgold_6tomobins.dat") # dimension = (6,2000) = (gg bin, ell bin)
    # LSSTxCIB
    clgCIB_matrix = np.loadtxt(spectra_path+"cl_CIBxLSSTgold_6tomobins.dat") # dimension = (6,2000) = (gg bin, ell bin)
    # CIB auto
    CIBauto = np.loadtxt(spectra_path+"cl_CIBauto.dat") # a vector of length 2000, ellmin = 8, ellmax = 2007
    # CIB x kappa
    CIBxkappa = np.loadtxt(spectra_path+"cl_CIBxkappa.dat") # a vector of length 2000, ellmin = 8, ellmax = 2007
    # Clkk
    Clkk = np.loadtxt(spectra_path+"cl_kk.dat") # a vector of length 2000, ellmin = 8, ellmax = 2007
    # CIB auto
    clII = np.loadtxt(spectra_path+"cl_CIBauto.dat") # a vector of length 2000, ellmin = 8, ellmax = 2007
    # CIB x kappa
    clkI = np.loadtxt(spectra_path+"cl_CIBxkappa.dat") # a vector of length 2000, ellmin = 8, ellmax = 2007
    # Clkk
    clkk = np.loadtxt(spectra_path+"cl_kk.dat") # a vector of length 2000, ellmin = 8, ellmax = 2007

    #Load the theoretical reconstruction noise spectrum
    sensNow = '2'
    depNow = '0'
    Nls = np.loadtxt(spectra_path+'SO_nlkk_plot/Apr17_mv_nlkk_deproj'+depNow+'_SENS'+sensNow+'_fsky_'+'16000'+'_iterOn.csv')
    nlkk = np.interp(ells, Nls[:,0], Nls[:,1])
    clkkrec = pad_cls(clkk) + nlkk #The theoretical phi reconstruction power spectrum

    num_of_tracers = 8
    cl = np.zeros((num_of_tracers,num_of_tracers, len(ells)), dtype='complex128') #Theory auto and cross spectra

    cl[0,0,:] = pad_cls(clkk)
    cl[1,1,:] = pad_cls(clgg_matrix[0,:])
    cl[2,2,:] = pad_cls(clgg_matrix[1,:])
    cl[3,3,:] = pad_cls(clgg_matrix[2,:])
    cl[4,4,:] = pad_cls(clgg_matrix[3,:])
    cl[5,5,:] = pad_cls(clgg_matrix[4,:])
    cl[6,6,:] = pad_cls(clgg_matrix[5,:])
    cl[7,7,:] = pad_cls(clII)

    cl[0,1,:] = cl[1,0,:] = pad_cls(clkg_matrix[0,:])
    cl[0,2,:] = cl[2,0,:] = pad_cls(clkg_matrix[1,:])
    cl[0,3,:] = cl[3,0,:] = pad_cls(clkg_matrix[2,:])
    cl[0,4,:] = cl[4,0,:] = pad_cls(clkg_matrix[3,:])
    cl[0,5,:] = cl[5,0,:] = pad_cls(clkg_matrix[4,:])
    cl[0,6,:] = cl[6,0,:] = pad_cls(clkg_matrix[5,:])
    cl[0,7,:] = cl[7,0,:] = pad_cls(clkI)

    cl[7,1,:] = cl[1,7,:] = pad_cls(clgCIB_matrix[0,:])
    cl[7,2,:] = cl[2,7,:] = pad_cls(clgCIB_matrix[1,:])
    cl[7,3,:] = cl[3,7,:] = pad_cls(clgCIB_matrix[2,:])
    cl[7,4,:] = cl[4,7,:] = pad_cls(clgCIB_matrix[3,:])
    cl[7,5,:] = cl[5,7,:] = pad_cls(clgCIB_matrix[4,:])
    cl[7,6,:] = cl[6,7,:] = pad_cls(clgCIB_matrix[5,:])

    spectra_matrix = cl.copy()
    spectra_matrix[0,0,:] = clkkrec

    # Calculate the weights and auxiliary spectra needed to generate Gaussian sims of individual tracers
    aux_cl, A = calculate_sim_weights(cl)

    # Draw harmonic coefficients from Gaussian distributions with the calculated auxiliary spectra
    a_alms = draw_gaussian_a_p(input_kappa_alm, aux_cl)

    # Combine weights and coefficients to generate sims of individual tracers
    tracer_alms = generate_individual_gaussian_tracers(a_alms, A, nlkk)

    # Calculate the optimal weights to form a multitracer map for delensing
    c_array = calculate_multitracer_weights(spectra_matrix, clkk)

    # Co-add the individual tracers using the weights we just calculated
    combined_kappa_alms = get_coadded_kappa_tracer(tracer_alms, c_array)

    # Convert phi to kappa and save
    combined_phi_alms = np.nan_to_num(hp.almxfl(combined_kappa_alms, 1/((1.0/2)*ells**2)))
    if save is True:
        np.save(output_path+'combined_phi_alms_simid'+str(simid), combined_phi_alms)
        return
    else:
        return combined_phi_alms

