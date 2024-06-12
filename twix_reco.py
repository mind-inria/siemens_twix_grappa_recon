import os
import torch
import numpy as np
import scipy
#import logging


from grappaND import GRAPPA_Recon
from twix_reader import read_twix_datafile
from twix_utils import fixShapeAndIFFT
from utils import rss


def _performNoiseDecorr(data, noise):
    R = np.cov(noise, rowvar=False)
    mean_abs_diag = np.mean(np.abs(np.diag(R)))
    R = R / mean_abs_diag
    np.fill_diagonal(R, np.abs(np.diag(R)))
    R_inv = np.linalg.inv(R)
    R_inv_sqrt = scipy.linalg.sqrtm(R_inv).astype(np.complex64)

    D = torch.from_numpy(R_inv_sqrt)

    data_sz = data.shape
    data_decorr = D @ data.reshape(data.shape[0], -1)
    data_decorr = data_decorr.reshape(data_sz)

    return data_decorr


def Twix_GRAPPA_Recon(filepath, savepath=None, performRSS=True, verbose=False):
    if not savepath or os.path.isdir(savepath):
        base = os.path.splitext(os.path.basename(filepath))[0]
        savepath = os.path.join(os.path.dirname(filepath) if not savepath else savepath, base + "_GRAPPArecon")
        
    if verbose:
        print(f"Input file: {filepath}")
        print(f"Output file: {savepath}" + ".npy\n")

    data, scan_info = read_twix_datafile(filepath)
    data['image'] = _performNoiseDecorr(data['image'], data['noise'])

    af = []
    if ('sPat', 'lAccelFact3D') in scan_info.hdr.MeasYaps:
        af.append(int(scan_info.hdr.MeasYaps[('sPat', 'lAccelFact3D')]))
    if ('sPat', 'lAccelFactPE') in scan_info.hdr.MeasYaps:
        af.append(int(scan_info.hdr.MeasYaps[('sPat', 'lAccelFactPE')]))

    rec = GRAPPA_Recon(data['image'], data['refscan'], af=af)
    if verbose: print("GRAPPA Reconstruction Done!")

    del data
    rec = rec.permute(0,3,1,2)

    rec = rec.numpy()
    
    if verbose: print("Fixing shape and IFFT..")
    rec = fixShapeAndIFFT(rec, scan_info)

    if performRSS:
        if verbose: print("RSS..")
        rec = rss(rec, axis=0)
        
    if verbose: print("Saving reconstruction..")
    np.save(savepath, rec)


if __name__ == "__main__":
    import sys
    Twix_GRAPPA_Recon(sys.argv[1], savepath=sys.argv[2] if len(sys.argv) > 2 else None)