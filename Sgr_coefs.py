import os
import gc
import numpy as np
import pynbody
import load_utils

import pyEXP

sgrBool = True
coldBool = False
vec = None
rng = None

sim_name = load_utils.simName(sgrBool,coldBool)
simPath = load_utils.simPath(sgrBool,coldBool)

storePath = "./coef_next/"

halo_config = """
id: sphereSL
parameters:
  numr : 4000
  rmin : 0.05
  rmax : 150.0
  Lmax : 4
  nmax : 4
  rmapping : 3.4
  modelname : Sgr_empirical.txt
  cachename : Sgr_empirical.cache
"""
halo_basis = pyEXP.basis.Basis.factory(halo_config)

snapVec = np.arange(0,713,1)

for i in snapVec:

    print(i)
    t = i/100
    
    massFactor = 1
    if not rng is None:
        massFactor = rng

    h = load_utils.loadSnap(simPath, i, sgr = sgrBool, cold = coldBool, dm = True, stars = True, bulge = True, rng=rng)

    temp = pynbody.analysis.halo.center(h[1], mode='ssc', retcen=True)
    # record the displacement at every time step
    if vec is None:
        vec = temp
    else:
        vec = np.vstack((vec, temp))
        
    # find sgr center
    
    
    # make the coefficients
    compname = 'Halo'

    # if you want to use the array creator, do this:
    mass = np.array(h[1].dark['mass'])*1e10
    if not rng is None:
        mass = mass/rng
    pos = np.array(h[1].dark['pos'])
    halo_coef = halo_basis.createFromArray(mass/massFactor,pos, time=t, center=temp)
    
    halo_coefs = pyEXP.coefs.Coefs.makecoefs(halo_coef, compname)
    halo_coefs.add(halo_coef)

    if os.path.isfile(storePath + 'halo/outcoef.'+sim_name+'_halo_{:03d}.h5'.format(i)):
        os.remove(storePath + 'halo/outcoef.'+sim_name+'_halo_{:03d}.h5'.format(i)) # removes the existing file
    halo_coefs.WriteH5Coefs(storePath + 'halo/outcoef.'+sim_name+'_halo_{:03d}.h5'.format(i))

    del(h)
    gc.collect()
np.save('vec_sgr.npy', vec)
