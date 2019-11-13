import numpy as np
import healpy as hp
import pickle
from matplotlib.pyplot import *
import prjlib
import curvedsky

pid, fid, r = prjlib.analysis_init(t='id',freq='145',ntype='cv')
pla, fla, r = prjlib.analysis_init(t='la',freq='coadd')
psa, fsa, r = prjlib.analysis_init(t='sa',freq='coadd')
pco, fco, r = prjlib.analysis_init(t='co',freq='coadd')
#psa, fsa, r = prjlib.analysis_init(t='sa',freq='145')
Dir = '../../data/sobs/delens/tmp/'
lcut = 2048

#Wla, w2, w4 = prjlib.window(fla.cmb.mask)
#Wsa, w2, w4 = prjlib.window(fsa.cmb.mask)
#Msa = Wsa/(Wsa+1e-30)

vmin, vmax = -1e-6, 1e-6
#vmin, vmax = -2e-7, 2e-7

#wla = hp.pixelfunc.ud_grade(Wla,psa.nside)
#Mla = wla/(wla+1e-30)
#iW = Wsa*Mla/(wla+1e-30)
#hp.mollview(Wsa+wla)
#show()


'''
for t in ['la','co']:
    Balm = pickle.load(open(Dir+'balm_1_'+t+'.pkl',"rb"))
    Q, U = Wsa*curvedsky.utils.hp_alm2map_spin(psa.npix,pid.lmax,pid.lmax,2,0*Balm,Balm)
    hp.mollview(Q,min=vmin,max=vmax)
    savefig('fig_bsat_'+t+'.png')
'''

#Ealm, Balm = pickle.load(open(fco.cmb.oalm['E'][1],"rb")), pickle.load(open(fco.cmb.oalm['B'][1],"rb"))
#Q, U = curvedsky.utils.hp_alm2map_spin(psa.npix,psa.lmax,psa.lmax,2,Ealm,Balm)

#Qs = hp.fitsfunc.read_map(fsa.cmb.lcdm[1],field=1)/2.72e6
#Qn = hp.fitsfunc.read_map(fsa.cmb.nois[1],field=1)/2.72e6
#Q = Wsa*(Qs+Qn)

#wElm, wBlm = pickle.load(open(Dir+'1_enco.pkl',"rb"))
#Q, U = curvedsky.utils.hp_alm2map_spin(psa.npix,lcut,lcut,2,wElm,wBlm)

#hp.mollview(Q)
#hp.mollview(Q,min=vmin,max=vmax)
#savefig('fig.png')
#show()

f1 = 'dalm_1_co_wiener'
f2 = 'wdalm_1_co_wiener'

for fi in [f1,f2]:
    Blm = pickle.load(open(Dir+fi+'.pkl',"rb"))
    Q, U = curvedsky.utils.hp_alm2map_spin(psa.npix,lcut,lcut,2,0*Blm,Blm)
    hp.mollview(Q,min=vmin,max=vmax)
    savefig('fig_'+fi+'.png')
    clf()

