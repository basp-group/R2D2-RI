# Requirements

1. Casacore https://github.com/casacore/casacore
2. Meqtrees https://github.com/ratt-ru/meqtrees/wiki/Installation

# General notes
1. Data extraction is performed using `pyxis`. The task should be performed in `./pyxisMs2mat` which contains the python script `pyxis_ms2mat.py`.
2. Extracted .mat file is saved in `../data/`. The file contain the following fields:
``` matlab
   "frequency"       % channel frequency
   "y"               % data (Stokes I)
   "u"               % u coordinate (in units of the wavelength)
   "v"               % v coordinate (in units of the wavelength)
   "w"               % w coordinate (in units of the wavelength)
   "nW"              % inverse of the noise standard deviation 
   "nWimag"          % square root of the imaging weights if available (Briggs or uniform), empty otherwise
   "maxProjBaseline" % maximum projected baseline (in units of the wavelength)
   ```    
3. The function in `pyxis_ms2mat.py` is an example. The user is encouraged to tailor it if the measurement set is organised differently.

# Example
Extracting (monochromatic) data at the frequency channel  `0` of the source with field ID `0`.

The user must provide the name/path to the measurement set `$MS`. The following inputs are optional.
```bash
$SRCNAME  # default SRCNAME="" : source nametag which defines the main directory of the extracted data. 
$FIELDID  # default FIELDID=0 : field ID of the source of interest, 
$FREQID   # default FREQID=0 : ID of the channnel to be extracted, 
```

From the terminal launch:
```bash
pyxis  MS=myms.ms SRCNAME=3c353 FIELDID=0 FREQID=0 getdata_ms
```

Data will be saved as .mat files in the directory `../data/`. The outcome is as follows
```bash
../data/3c353_data_ch_1.mat
```
