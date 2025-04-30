import glob
from astropy.io import fits
from scipy.io import savemat
import multiprocessing as mp
import numpy as np
import argparse
import os
from tqdm import tqdm

def get_psf_peak_sidelobe_from_psf_file(psf_file, output_file):
    psf = abs(fits.getdata(psf_file))
    psf_peak_1sl = get_psf_peak_sidelobe(psf)
    savemat(output_file, {"psf_peak_1st_sidelobe": psf_peak_1sl})

def get_psf_peak_sidelobe(psf):
    psf = psf.squeeze()
    psf_size = psf.shape
    central_pixel = int(psf_size[0] // 2)
    psf_c_row = psf[central_pixel, :]  # central row of the PSF
    psf_c_col = psf[:, central_pixel]  # central column of the PSF

    # row
    found = False
    increasing = False
    decreasing = False
    cur_pixel_pos_row = central_pixel
    while not found:
        next_pixel = cur_pixel_pos_row + 1
        if psf_c_row[next_pixel] > psf_c_row[cur_pixel_pos_row]:
            increasing = True
        if psf_c_row[next_pixel] < psf_c_row[cur_pixel_pos_row] and increasing:
            decreasing = True
        if increasing and decreasing:
            found = True
        else:
            cur_pixel_pos_row += 1
    cur_pixel_neg_row = central_pixel
    increasing = False
    decreasing = False
    found = False
    while not found:
        next_pixel = cur_pixel_neg_row - 1
        if psf_c_row[next_pixel] > psf_c_row[cur_pixel_neg_row]:
            increasing = True
        if psf_c_row[next_pixel] < psf_c_row[cur_pixel_neg_row] and increasing:
            decreasing = True
        # print(f"pixel {cur_pixel_neg_row}: increasing {increasing}, decreasing {decreasing}")
        if increasing and decreasing:
            found = True
        else:
            cur_pixel_neg_row += 1

    # col
    found = False
    increasing = False
    decreasing = False
    cur_pixel_pos_col = central_pixel
    while not found:
        next_pixel = cur_pixel_pos_col + 1
        if psf_c_col[next_pixel] > psf_c_col[cur_pixel_pos_col]:
            increasing = True
        if psf_c_col[next_pixel] < psf_c_col[cur_pixel_pos_col] and increasing:
            decreasing = True
        if increasing and decreasing:
            found = True
        else:
            cur_pixel_pos_col += 1
    cur_pixel_neg_col = central_pixel
    increasing = False
    decreasing = False
    found = False
    while not found:
        next_pixel = cur_pixel_neg_col - 1
        if psf_c_col[next_pixel] > psf_c_col[cur_pixel_neg_col]:
            increasing = True
        if psf_c_col[next_pixel] < psf_c_col[cur_pixel_neg_col] and increasing:
            decreasing = True
        # print(f"pixel {cur_pixel_neg_col}: increasing {increasing}, decreasing {decreasing}")
        if increasing and decreasing:
            found = True
        else:
            cur_pixel_neg_col += 1
    psf_peak_1sl = max(
        [
            psf_c_row[cur_pixel_neg_row],
            psf_c_row[cur_pixel_pos_row],
            psf_c_col[cur_pixel_neg_col],
            psf_c_col[cur_pixel_pos_col],
        ]
    )
    return psf_peak_1sl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--psf_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--ncpus", type=int)
    args = parser.parse_args()
    psf_files = glob.glob(os.path.join(args.psf_dir, "*.fits"))
    print(f"Found {len(psf_files)} PSF files")
    # output_files = [f.replace(".fits", ".mat") for f in psf_files]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_files = [os.path.join(args.output_dir, os.path.basename(f).replace(".fits", ".mat")) for f in psf_files]
    if args.ncpus > 1:
        print(f"Using multiprocessing with {args.ncpus} cores")
        with mp.Pool(args.ncpus) as pool:
            pool.starmap(get_psf_peak_sidelobe, tqdm(zip(psf_files, output_files), total=len(psf_files)))
    else:
        for psf_file, output_file in tqdm(zip(psf_files, output_files)):
            get_psf_peak_sidelobe(psf_file, output_file)
        