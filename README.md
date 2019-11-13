# Pipeline for SO delensing study (L3.3)

This package contains Python codes for constructing B-mode lensing template from cosmic microwave background anisotropies (CMB) in curvedsky. 

# Dependencies

The current code depends on the following python modules:

  - numpy, healpy, pickle, matplotlib, configparser, sys
  - mapsims (https://github.com/simonsobs/mapsims)
  - cmblensplus (https://toshiyan.github.io/clpdoc/html/)

# Files

The main files for the analysis are as follows:

  - simmap.py: generate simulated signal and noise maps for each frequency and patch
  - cmb_map2alm.py: spherical harmonic transform of map to alm at each frequency and patch, and compute power spectra
  - cmb_wiener.py: Optimally-combined alms from LAT and SAT with Wiener filtering.
  - delens.py: construct lensing B-mode template and compute auto and cross spectra between lensing template and SAT B-modes
  - gen_mass_sims.py: generate random Gaussian fields from the covariance of the mass-tracers. 
  - prjlib.py: read and define parameters and filename for analysis and some functions
  - params.ini: parameter file

# Contacts

  Toshiya Namikawa (namikawa at slac.stanford.edu)
  Anton Baleato (a.baleatolizancos@ast.cam.ac.uk)

