## Rodrigues compatibale simulation pipeline
## Script based on a template provided by Sphesihle Makhathini [sphemakh@gmail.com]


import math
import os
import im
import im.argo
import imager
import ms
import numpy as np
import pyrap.tables
import Pyxis
import scipy.io as sio
from Pyxis.ModSupport import *

PI = math.pi
FWHM = math.sqrt(math.log(256))
C = 299792458

def getdata_ms( msfile="$MS", srcname="$SRCNAME", srcid="$FIELDID", freqid="$FREQID"):
    """Pyxis function to be used from the command line to extract data from a single MS.
    ----------
    msfile : str, required
        Path to the MS.
    srcname : str, optional
        Name of the target source, by default "".
    srcid : str, optional
        Field id of the target source, by default 0.
    freqid : str, optional
        Index of the channel frequency to extract by default 0.
    
    """    
    
    msname = II("%s" % msfile)
    try:
        srcname = II("%s" % srcname)
    except:
        srcname =""
    try:
        srcid = int(II("%s" % srcid))
    except: 
        srcid =0
        info("Field ID is not provided, data of field ID 0 will be extracted")
    try:
        freqid = int(II("%s" % freqid))
    except:
        freqid = 0

    data_dir = "../data"
    x.sh("mkdir -p  %s" % data_dir)
    dataFileName = "%s/%s_data_ch_%s.mat" %(data_dir, srcname, freqid + 1)
    info("Data .mat file will be saved as:  %s" %dataFileName)

    info("MS: %s"%msname)
    tab = ms.ms(msname, write=False)
    info("MS table columns:", *(tab.colnames()))

    spwtab = ms.ms(msname, subtable="SPECTRAL_WINDOW")    
    freqsVect = spwtab.getcol("CHAN_FREQ")
           
    # load remaining specs
    field = tab.getcol("FIELD_ID")
    srcrows = field == srcid

    # u,v,w,
    uvw = tab.getcol("UVW")
    uvw = uvw[srcrows, :]
    nmeas = len(uvw[:, 0])
    info("Number of measurements per channel: %s" % nmeas)

    # flag row (common across channels)
    flag_row = tab.getcol("FLAG_ROW")
    flag_row = flag_row[srcrows]
    flag_row = flag_row.transpose()
    flag_row = flag_row.astype(float)  
    npts = len(flag_row)
    info( "Number of measurements per channel associated with the target source: %s" %npts)

    # natural weights
    try:
        weight_ch = tab.getcol("WEIGHT_SPECTRUM")
        weight_ch_shape = weight_ch.shape
        ncorr = weight_ch_shape[2]
        weight_ch = weight_ch[srcrows, freqid, :]
        w1 = weight_ch[:, 0]
        w4 = weight_ch[:, ncorr - 1]
    except:
        info("WEIGHT_SPECTRUM not available --> using WEIGHT")
        weight = tab.getcol("WEIGHT")
        weight_shape = weight.shape
        ncorr = weight_shape[1]
        weight = weight[srcrows,:]
        w1 = weight[:, 0]
        w4 = weight[:, ncorr -1]

    # load data
    try:
        data_full = tab.getcol("CORRECTED_DATA")
    except:
        data_full = tab.getcol("DATA")
        info("CORRECTED_DATA not found, reading DATA instead ")
    data_full = data_full[srcrows, freqid, :]
    data_corr_1 = data_full[:, 0]
    data_corr_4 = data_full[:, ncorr - 1]

    # Stokes I & associated natural weights
    data = ( w1 * data_corr_1 + w4 * data_corr_4 ) / (w1 + w4)
    weight_natural = w1 + w4

    # load remaining flags
    flagAll = tab.getcol("FLAG")
    flagAll = flagAll[srcrows, freqid, :]
    flagAll = flagAll.astype(float)
    flagAll = flagAll[:, 0] + flagAll[:, ncorr-1]

    # briggs/uniform imaging weights
    weight_imaging = ""
    try:
        weight_imaging = tab.getcol("IMAGING_WEIGHT")
        weight_imaging = weight_imaging[srcrows, 0]
        # typically same weights for all corr
        info("IMAGING_WEIGHTS is  available")
    except:
        try:
            weight_imaging = tab.getcol("IMAGING_WEIGHT_SPECTRUM")
            weight_imaging = weight_imaging[srcrows, freqid, 0]
            # typically same weights for all corr
            info("IMAGING_WEIGHTS_SPECTRUM is available")
        except:
            info("IMAGING WEIGHTS not found")

    # flag
    flag = ((np.absolute(data) ==False).astype(float)+flagAll + flag_row)
    flag = (flag==False)
  
    # applying flags 
    frequency = freqsVect[0,freqid]
    y = data[flag]
    u = uvw[flag, 0]/ (C / frequency)
    v = uvw[flag, 1]/ (C / frequency)
    w = uvw[flag, 2]/ (C / frequency)
    nW = np.sqrt(weight_natural[flag])
    try:
         nWimag = np.sqrt(weight_imaging[flag])
    except:
         nWimag =[]

    # maximum projected baseline
    maxProjBaseline = (np.sqrt(max(u ** 2 + v ** 2))).astype(float)

    # save data
    info("Reading data and writing file..Freq %s" %freqid)
    sio.savemat(
                dataFileName,
                {
                    "frequency": frequency,
                    "y": y,# data (Stokes I)
                    "u": u,# u coordinate (in units of the wavelength)
                    "v": v, # v coordinate (in units of the wavelength)
                    "w": w, # w coordinate  (in units of the wavelength)              
                    "nW": nW, # 1/sigma: square root of natural weights
                    "nWimag": nWimag,  # square root of the imaging weights if available (Briggs or uniform)
                    "maxProjBaseline": maxProjBaseline,#max projected baseline 
                },
            )
    tab.close()


