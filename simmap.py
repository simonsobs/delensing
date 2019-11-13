import numpy as np
import healpy as hp
from mapsims import SONoiseSimulator
from mapsims import SOStandalonePrecomputedCMB
from mapsims import Channel
import prjlib


def sim_cmb(fmap,telescope,nside,snmin,snmax,band):
    ch  = Channel(telescope=telescope,band=band)
    for i in range(snmin,snmax):
        print(i)
        sim = SOStandalonePrecomputedCMB(i,nside=nside,input_units='uK_CMB')
        map = SOStandalonePrecomputedCMB.simulate(sim,ch)
        hp.fitsfunc.write_map(fmap[i],map,overwrite=True)


def sim_noise(fmap,telescope,nside,snmin,snmax,band):
    ch  = Channel(telescope=telescope,band=band)
    for i in range(snmin,snmax):
        print(i)
        sim = SONoiseSimulator(nside,apply_beam_correction=False)
        print('simulating')
        map = SONoiseSimulator.simulate(sim,ch)
        hp.fitsfunc.write_map(fmap[i],map,overwrite=True)


if __name__ == '__main__':

    for freq in ['93','145']:
        for t in ['sa','la']:
            p, f, r = prjlib.analysis_init(t=t,freq=freq)
            band = int(p.freq)
            sim_cmb(f.cmb.lcdm,str.upper(t),p.nside,p.snmin,p.snmax,band)
            sim_noise(f.cmb.nois,str.upper(t),p.nside,p.snmin,p.snmax,band)

