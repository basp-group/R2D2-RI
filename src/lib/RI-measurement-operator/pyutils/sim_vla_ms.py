# scripts to create Fourier sampling patterns in .mat files containing the fields:
# `u`,`v`,`w` (all in units of meter), `frequency` (MHz), `nominal_pixelsize` (arcsec)
# Author: A. Dabbech

from simms import simms
from casatools import table
from casatasks import concat
import numpy as np
import scipy.io as sio
import os
import argparse
import math
import timeit
import matplotlib.pyplot as plt

# constants
c = 299792458  # Speed of light

# set dirs
mydir = os.getcwd()
vlasimsdir = mydir + "/vla_sims/"
os.system("mkdir %s" % vlasimsdir)
msdir = vlasimsdir + "ms/"
os.system("mkdir %s" % msdir)
uvdir = vlasimsdir + "uvw/"
os.system("mkdir %s" % uvdir)
pngdir = vlasimsdir + "png/"
os.system("mkdir %s" % pngdir)

## user-input
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--npatterns', default=1)
args = parser.parse_args()

def main():
    # number of sampling patterns & MS tables to generate
    print('############# User-input:')
    npatterns = int(args.npatterns)
    print('Number of requested Fourier sampling patterns: %s' % npatterns)
    # generate sampling patterns aka uvw-coverages & related info
    for i in range(npatterns):
        start = timeit.default_timer()
        print('############# Fourier sampling pattern id: %s' % i)

        print('##### observation settings:')
        ## number of VLA antennas
        na = 28
        ## pointing direction: Right Ascension
        ra = np.random.uniform() * 23  # [0h,23h]
        ra_h = int(ra)
        ra_min = abs(int((ra - ra_h) * 59))
        msra = str(ra_h) + "h" + str(ra_min) + "m0s"
        print("info: (param) RA: %s" % msra)
        ## pointing direction: Declination
        dec_deg = int(np.random.uniform() * 55 + 5)  #  [5,60] #val dec: 34°04′43.497′′
        dec_min = int(np.random.uniform() * 59)
        msdec = str(dec_deg) + "d" + str(dec_min) + "m0s"
        print("info: (param) DEC: %s" % msdec)

        ## time specs
        dta_min = 5  # A config: min total observation time
        dta = 10 #np.random.uniform(5, 5 + dta_min)  # A config: total observation time in [5,10]
        print("info: (param) obs. time with config A of freqs: %.2f h" % dta)
        dtc_min = 1  # C  config: min total observation time
        dtc = np.random.uniform() * 2 + dtc_min  # C config: total observation time in [1,3]
        print("info: (param) obs. time with config C of freqs: %.2f h" % dtc)
        dt_step = 30  # integration time/ time-step (fixed but could be random)
        print("info: (param) time step: %.2f sec" % dt_step)
        ## freq specs
        freq0 = 1e9  # starting freq.
        freq_ratio = (np.random.uniform() + 1)  # in  [1,2]
        nfreqs_max = 4  # max number of freq. (feel free to change)
        freq_ratio_min = 1.1  # lower-bound  (could be changed)
        if freq_ratio > freq_ratio_min:
            nfreqs = int(max(1, (np.random.uniform() * nfreqs_max)))  # number of freqs
        else:
            nfreqs = 1
        freq_vect = np.linspace(freq0, freq0 * freq_ratio, num=nfreqs)  # freq. vect
        if nfreqs > 1:
            msdfreq = "%sMHz" % ((freq_vect[1] - freq_vect[0]) / 1e6)
        else:
            msdfreq = "1MHz"

        msfreq0 = "%sMHz" % (freq0 / 1e6)

        print("info: (param) number of freqs: %s" % nfreqs)
        if nfreqs > 1:
           print("info: (param) frequency ratio between the highest and the lowest:  %.2f" % freq_ratio)

        print('##### Augmentation:')
        ## rotation angle of the uv-coverage/psf
        rot_theta = np.random.uniform(0, 360)  # to switch off, set to 0
        print("info: (param) rotation angle of the sampling pattern: %.2f deg" % rot_theta)
        flag_percentage_max = 0.2  # upper-bound on the flagging percentage

        print('##### Imaging settings (required to generate imaging weights):')
        # briggs weighting: robust param
        briggs = 0  # np.random.uniform(-1, 1)  # B
        print("info: (param) briggs weigting: %.2f" % briggs)

        ## MS filenames
        # config A
        exta = 'id_' + str(i) + '_dec_' + str(dec_deg) + '_dta_' + str("%.2f" % dta)
        mymsfile_a = '%svlaa_%s.MS' % (msdir, exta)
        # config C
        extc = 'id_' + str(i) + '_dec_' + str(dec_deg) + '_dtc_' + str("%.2f" % dtc)
        mymsfile_c = '%svlac_%s.MS' % (msdir, extc)
        # ms file
        ext = 'id_' + str(i) + '_dt_' + str("%.2f" % (dta + dtc)) + "_freqratio_" + str(
            "%.2f" % freq_ratio) + '_nfreq_' + str(nfreqs) + '_rotation_' + str("%.2f" % rot_theta)
        mymsfile = '%svla_%s.MS' % (msdir, ext)

        print('##### Create empty MS: start')
        # delete old ms just in case
        try:
            os.system('rm -rf %s' % mymsfile)
        except:
            print('')

        # --------------------------------------
        # Step 1: Generate both MSs
        # --------------------------------------
        print("CASA:start ----------------------------")
        print('##### Create empty MS .. A config')
        # A config
        simms.create_empty_ms(msname=mymsfile_a, tel='vlaa', pos="%s/observatories/vlaa.itrf.txt" % mydir,
                              pos_type='ascii', coords="itrf",
                              synthesis=dta, dtime=dt_step, dfreq=msdfreq, freq0=msfreq0, nchan=[nfreqs],
                              ra=msra, dec=msdec, scan_length=[dta + 0.01], scan_lag=0,
                              stokes="XX YY", setlimits=True, optimise_start=True)
        # C config
        print('##### Create empty MS .. C config')
        simms.create_empty_ms(msname=mymsfile_c, tel='vlac', pos="%s/observatories/vlaa.itrf.txt" % mydir,
                              pos_type='ascii', coords="itrf",
                              synthesis=dtc, dtime=dt_step, dfreq=msdfreq, freq0=msfreq0, nchan=[nfreqs],
                              ra=msra, dec=msdec, scan_length=[dtc + 0.01], scan_lag=0,
                              stokes="XX YY", setlimits=True, optimise_start=True)

        print('##### Concat  MSs ..')
        concat(vis=[mymsfile_a, mymsfile_c], concatvis=mymsfile)
        os.system('rm -rf %s' % mymsfile_a)  # delete vla-a MS
        os.system('rm -rf %s' % mymsfile_c)  # delete vla-c MS
        print("CASA:end ----------------------------")
        # --------------------------------------
        # Step 2: Apply (random) flags & extract (final) uvw
        # --------------------------------------
        tb = table()
        tb.open(mymsfile, nomodify=False)
        print("MS table columns:", *(tb.colnames()))
        print('##### Random flagging & data extraction: start')

        # check number of scans
        scans = tb.getcol('SCAN_NUMBER')
        nscans = len(np.unique(scans))
        # get number of meas. per freq.
        nmeas = len(scans)
        print("info: number of scans %s" % nscans)

        # get uvw col
        uvw = tb.getcol('UVW')
        # apply rotation to the Fourier sampling pattern (aka uv-coverage)
        uvw_rot = np.zeros(uvw.shape)
        uvw_rot[0,:] = math.cos(math.radians(rot_theta)) * uvw[0,:] - math.sin(math.radians(rot_theta)) * uvw[1,:]
        uvw_rot[1,:] = math.sin(math.radians(rot_theta)) * uvw[0,:] + math.cos(math.radians(rot_theta)) * uvw[1,:]
        uvw_rot[2,:] = uvw[2,:]
        # overwrite uvw col
        tb.putcol('UVW', uvw_rot)

        # get FLAG col (set to False everywhere)
        flag = tb.getcol('FLAG')
        ## init
        u, v, w = [], [], []
        ## apply random flagging at each frequency & get uvw
        ref_freq = (freq_vect[nfreqs - 1] + freq_vect[0]) / 2

        for ifreq in range(len(freq_vect)):
            flag_percentage = np.random.uniform() * flag_percentage_max
            print("info: (param) freq %s: flagging percentage %.4f " % (ifreq, 100 * flag_percentage))
            flagged_rows = (np.random.choice(np.linspace(0, nmeas - 1, num=nmeas),
                                             size=math.floor(nmeas * flag_percentage))).astype(int)
            flag[:, ifreq, flagged_rows] = True
            select_rows = (flag[0, ifreq, :] == False)

            # freq ratio to be applied
            ifreq_ratio = freq_vect[ifreq] / ref_freq
            # get uvw after flagging (in meter)
            u.extend(uvw_rot[0, select_rows] / ifreq_ratio)
            v.extend(uvw_rot[1, select_rows] / ifreq_ratio)
            w.extend(uvw_rot[2, select_rows] / ifreq_ratio)

        print("info: initial number  meas. %s" % (nfreqs * nmeas))
        print("info: number of meas. after flagging %s" % len(u))
        # overwrite FLAG col & update MS
        tb.putcol('FLAG', flag)  # update flag column in the MS
        tb.close()
        print('##### Random flagging & data extraction: end')

        # reshape vars
        u = np.reshape(np.array(u), (len(u), 1))
        v = np.reshape(np.array(v), (len(u), 1))
        w = np.reshape(np.array(w), (len(u), 1))

        # info needed for pixelsize
        wavelength = c / ref_freq
        maxProjBaseline = (np.sqrt(max(u ** 2 + v ** 2))).astype(float) / wavelength
        print('info: frequency: %f GHz ' % (ref_freq / 1e9))
        print("info: max. projected baseline in units of the wavelength %s " % maxProjBaseline)

        # ------------------------------
        # Step 3: Save uvw to .mat
        # ------------------------------
        # nominal pixelsize (superresolution x1): upper bound on the pixel size used during imaging
        nominal_pixelsize = 1 / (2 * maxProjBaseline) * (180 / math.pi) * 3600
        print("info: nominal pixelsize (i.e. super-resolution x1):  %f arcsec" % nominal_pixelsize)

        # uvw saved in units of meter
        sampling_pattern_uvw = {'u': u, 'v': v, 'w': w, 'frequency': ref_freq,
                                'nominal_pixelsize': nominal_pixelsize}

        # final  mat file
        uvwmatfile = uvdir + 'uvw_' + ext + '.mat'
        print("info: saving .mat file: %s" % uvwmatfile)
        sio.savemat(uvwmatfile, sampling_pattern_uvw)

        # additional plot of the uv-coverage (for info only)
        plt.figure()
        plt.scatter(u, v, color='red', s=0.01)
        plt.scatter(-u, -v, color='blue', s=0.01)
        plt.title("dt:" + str("%.1f" % (dta + dtc)) + ", ra:" + str("%.2f" % ra) + ", dec:" + str("%.2f" % dec_deg)
                  + ', rot:' + str("%.1f" % rot_theta) + ', nfreqs:' + str("%.2f" % nfreqs) + ', freq ratio:' + str(
            "%.2f" % freq_ratio))
        plt.savefig(pngdir + 'uv_' + ext + '.png')
        plt.close()
        print(
            "info: mat file created successfully, which includes these fields:  `u`,`v`,`w` (all in units of meter), `frequency` (MHz), `nominal_pixelsize` (arcsec)")
        print('##### File saved.')

    ## delete tmp dirs
    # os.system("rm -rf %s" % mymsfile)

    stop = timeit.default_timer()

    print('Time to generate sampling patterns & files: %2.f sec ' % (stop - start))
    print('##### END.')

if __name__ == "__main__":
    main()
