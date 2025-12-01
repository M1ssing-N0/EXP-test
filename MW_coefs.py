import os
import gc
import numpy as np
import pandas as pd
import pynbody
import load_utils
import time

import pyEXP

sgrBool = True
coldBool = False

rng = None

sim_name = load_utils.simName(sgrBool,coldBool)
simPath = load_utils.simPath(sgrBool,coldBool)

storePath = "./coef_next/"

halo_config = """
id: sphereSL
parameters:
  numr : 4000
  rmin : 0.1
  rmax : 500.0
  Lmax : 6
  nmax : 4
  rmapping : 52
  modelname : MW_halo_empirical.txt
  cachename : MW_halo_empirical.cache
"""
halo_basis = pyEXP.basis.Basis.factory(halo_config)

bulge_config = """
id         : sphereSL
parameters :
  numr     : 4000
  rmin     : 0.01
  rmax     : 1000.0 #1000.0 Check max distance
  Lmax     : 0
  nmax     : 10
  rmapping  : 1 #52
  modelname : MW_bulge_empirical.txt
  cachename : MW_bulge.empirical.cache
"""
bulge_basis = pyEXP.basis.Basis.factory(bulge_config)

config_disk_final = """
id: cylinder
parameters:
  acyl: 3.5
  hcyl: 2
  mmax: 2
  rcylmin: 0.001
  rcylmax: 20.0
  nmax: 32
  nmaxfid: 48
  ncylnx: 512
  ncylny: 256
  lmaxfid: 96
  ncylodd: 12
  rnum: 200
  pnum: 1
  tnum: 80
  logr: true
  cachename: eof.cache.teng.file
"""
disk_basis = pyEXP.basis.Basis.factory(config_disk_final)

snapVec = np.arange(0,649,1)

for i in snapVec:
    print('##########################')
    print(f'running coefs of {i}')
    print('##########################')
    t = i/100
    
    massFactor = 1
    if not rng is None:
        massFactor = rng

    h = load_utils.loadSnap(simPath, i, sgr = sgrBool, cold = coldBool, dm = True, stars = True, bulge = True, rng=rng)
    
    temp = pynbody.analysis.halo.center(h[0], mode='ssc', retcen=True)

    # make the coefficients
    compname = 'Halo'

    # if you want to use the array creator, do this:
    mass = np.array(h[0].dark['mass'])*1e10
    if not rng is None:
        mass = mass/rng
    pos = np.array(h[0].dark['pos'])
    halo_coef = halo_basis.createFromArray(mass/massFactor, pos, time=t, center=temp)
    
    halo_coefs = pyEXP.coefs.Coefs.makecoefs(halo_coef, compname)
    halo_coefs.add(halo_coef)

    if os.path.isfile(storePath + 'halo/outcoef.'+sim_name+'_halo_{:03d}.h5'.format(i)):
        os.remove(storePath + 'halo/outcoef.'+sim_name+'_halo_{:03d}.h5'.format(i)) # removes the existing file
    halo_coefs.WriteH5Coefs(storePath + 'halo/outcoef.'+sim_name+'_halo_{:03d}.h5'.format(i))
    
    
    # make the coefficients
    compname = 'Bulge'

    # if you want to use the array creator, do this:
    mass = np.array(h[0].gas['mass'])*1e10
    if not rng is None:
        mass = mass/rng
    pos = np.array(h[0].gas['pos'])
    bulge_coef = bulge_basis.createFromArray(mass/massFactor,pos, time=t, center=temp)

    bulge_coefs = pyEXP.coefs.Coefs.makecoefs(bulge_coef, compname)
    bulge_coefs.add(bulge_coef)

    if os.path.isfile(storePath + 'bulge/outcoef.'+sim_name+'_bulge_{:03d}.h5'.format(i)):
        os.remove(storePath + 'bulge/outcoef.'+sim_name+'_bulge_{:03d}.h5'.format(i)) # removes the existing file
    bulge_coefs.WriteH5Coefs(storePath + 'bulge/outcoef.'+sim_name+'_bulge_{:03d}.h5'.format(i))

    compname = 'Disk'
    mass = np.array(h[0].stars['mass'])*1e10
    if not rng is None:
        mass = mass/rng
    pos = np.array(h[0].stars['pos'])
    disk_coef = disk_basis.createFromArray(mass/massFactor,pos, time=t, center=temp)

    disk_coefs = pyEXP.coefs.Coefs.makecoefs(disk_coef, compname) # it just creates the memory of the coeffs
    disk_coefs.add(disk_coef)

    if os.path.isfile(storePath + 'disk/outcoef.'+sim_name+'_disk_{:03d}.h5'.format(i)):
        os.remove(storePath + 'disk/outcoef.'+sim_name+'_disk_{:03d}.h5'.format(i)) # removes the existing file
    disk_coefs.WriteH5Coefs(storePath + 'disk/outcoef.'+sim_name+'_disk_{:03d}.h5'.format(i))


    del(h)
    gc.collect()
