# script to extra data and relevant information in .mat file
# Author: A. Dabbech
import os
import sys
import argparse
import numpy as np
import scipy.io as sio
import math
from casacore import tables

# constants
c = 299792458

# set dirs
mydir = os.getcwd()
data_dir = mydir + "/data/"
os.system("mkdir -p %s" % data_dir)


def main():
    parser = argparse.ArgumentParser(description='Extract data file from MS')
    parser.add_argument('--msfile', type=str, default=None, help='Path to MS file')
    parser.add_argument('--srcname', type=str, default='', help='Source name')
    parser.add_argument('--srcid', type=int, default=0, help='Source/Field ID')
    parser.add_argument('--freqid', type=int, default=0, help='Frequency ID')
    args = parser.parse_args()
    msfile =  args.msfile
    srcname = args.srcname
    srcid = args.srcid
    freqid = args.freqid

    # get data
    tab = tables.table(msfile)
    print("INFO: MS table columns:", *(tab.colnames()))
    print("INFO: Reading data ..Freq %s" % freqid)
    ## freq : table & freq. channels
    spwtab = tables.table("%s/SPECTRAL_WINDOW"%msfile)
    freqs = spwtab.getcol("CHAN_FREQ")

    ## load remaining specs
    # identify meas. with the source ID
    field = tab.getcol("FIELD_ID")
    srcrows = field == srcid
    # number of meas.
    nmeas = len(srcrows)
    print("INFO: Number of measurements per channel associated with the target source: %s" % nmeas)

    ## natural weights: noise vect:  1/variance
    try:
        weight_ch = tab.getcol("WEIGHT_SPECTRUM")
        # identify number of avail. corr. (2 or 4)
        weight_ch_shape = weight_ch.shape
        ncorr = weight_ch_shape[2]
        # select relevant rows
        weight_ch = weight_ch[srcrows, freqid, :]
        w1 = weight_ch[:, 0]
        w4 = weight_ch[:, ncorr - 1]
    except:
        print("INFO: WEIGHT_SPECTRUM not available, will use WEIGHT")
        weight = tab.getcol("WEIGHT")
        # identify number of avail. corr. (2 or 4)
        weight_shape = weight.shape
        ncorr = weight_shape[1]
        # select relevant rows
        weight = weight[srcrows, :]
        w1 = weight[:, 0]
        w4 = weight[:, ncorr - 1]

    ## data
    try:
        data = tab.getcol("CORRECTED_DATA")
    except:
        data = tab.getcol("DATA")
        print("INFO: CORRECTED_DATA not found, reading DATA instead ")
    data = data[srcrows, freqid, :]
    # Stokes I & associated natural weights
    data = (w1 * data[:, 0] + w4 * data[:, ncorr - 1]) / (w1 + w4)
    data = np.reshape(np.array(data), (nmeas, 1))
    weight_natural = np.reshape(np.array(w1 + w4), (nmeas, 1))

    ## flag
    # flag row (common across channels)
    flag_row = tab.getcol("FLAG_ROW")
    flag_row = (np.reshape(np.array(flag_row[srcrows]), (nmeas,))).astype(float)
    # load remaining flags
    flag = tab.getcol("FLAG")
    flag = (flag[srcrows, freqid, :]).astype(float)
    flag = np.reshape(np.array(flag[:, 0] + flag[:, ncorr - 1]), (nmeas,))
    # combine all flags
    flag_data = np.reshape((np.absolute(data) == False).astype(float), (nmeas,))
    flag = (flag + flag_data + flag_row) == False
    flag = np.reshape(np.array(flag), (nmeas,))

    ## u,v,w,
    uvw = tab.getcol("UVW")
    uvw = uvw[srcrows, :]

    ## briggs/uniform imaging weights
    weight_imaging = []
    try:
        weight_imaging = tab.getcol("IMAGING_WEIGHT")
        weight_imaging = weight_imaging[srcrows, 0]
        # typically same weights for all corr
        print("INFO: IMAGING_WEIGHTS is  available")
    except:
        try:
            weight_imaging = tab.getcol("IMAGING_WEIGHT_SPECTRUM")
            weight_imaging = weight_imaging[srcrows, freqid, 0]
            # typically same weights for all corr
            print("INFO: IMAGING_WEIGHTS_SPECTRUM is available")
        except:
            print("INFO: IMAGING WEIGHTS not found")
    tab.close()

    ## applying flags
    frequency = freqs[0, freqid]
    y = data[flag]
    nmeasflag = len(y)
    u = np.reshape(uvw[flag, 0] / (c / frequency), (nmeasflag, 1))
    v = np.reshape(uvw[flag, 1] / (c / frequency), (nmeasflag, 1))
    w = np.reshape(uvw[flag, 2] / (c / frequency), (nmeasflag, 1))
    nW = (np.sqrt(weight_natural[flag]))
    try:
        nWimag = (np.sqrt(weight_imaging[flag]))
        nWimag = np.reshape(np.array(nWimag), (nmeasflag, 1))
    except:
        nWimag = []

    ## maximum projected baseline (used for pixel size)
    maxProjBaseline = (np.sqrt(max(u ** 2 + v ** 2))).astype(float)

    ## save data
    datamatfile = "%s/%s_data_ch_%s.mat" % (data_dir, srcname, freqid + 1)
    print("INFO: Saving data ..Freq %s" % freqid)


    sio.savemat(
        datamatfile,
        {
            "frequency": frequency,
            "y": y,  # data (Stokes I)
            "u": u,  # u coordinate (in units of the wavelength)
            "v": v,  # v coordinate (in units of the wavelength)
            "w": w,  # w coordinate  (in units of the wavelength)
            "nW": nW,  # 1/sigma: square root of natural weights
            "nWimag": nWimag,  # square root of the imaging weights if available (Briggs or uniform)
            "maxProjBaseline": maxProjBaseline,  # max projected baseline  (in units of the wavelength)
        },
    )
    print("INFO: Data .mat file saved:  %s" % datamatfile)
    print("DONE.")


if __name__ == "__main__":
    main()
