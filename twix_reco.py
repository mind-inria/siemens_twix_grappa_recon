import os
import numpy as np

from grappaND2 import GRAPPA_Recon
from twix_reader import read_twix_datafile
from twix_utils import fixShapeAndIFFT, performNoiseDecorr
from utils import rss, save_nifti


def Twix_GRAPPA_Recon(filepath,
                      savepath=None,
                      performRSS=True,
                      save_npy=True,
                      save_nii=True,
                      verbose=True
) -> None:
    
    if not savepath or os.path.isdir(savepath):
        base = os.path.splitext(os.path.basename(filepath))[0]
        savepath = os.path.join(os.path.dirname(filepath) if not savepath else savepath, base + "_GRAPPArecon")
        
    if verbose:
        print(f"Input file: {filepath}")
        print(f"Output file: {savepath}" + ".npy\n")
        print(f"Output file: {savepath}" + ".npy\n")

    data, scan_info = read_twix_datafile(filepath)
    data['image'] = performNoiseDecorr(data['image'], data['noise'])

    af = []
    if ('sPat', 'lAccelFactPE') in scan_info.hdr.MeasYaps:
        af.append(int(scan_info.hdr.MeasYaps[('sPat', 'lAccelFactPE')]))
    if ('sPat', 'lAccelFact3D') in scan_info.hdr.MeasYaps:
        af.append(int(scan_info.hdr.MeasYaps[('sPat', 'lAccelFact3D')]))

    if verbose: print("GRAPPA Reconstruction started...")
    rec = GRAPPA_Recon(data['image'], data['refscan'], af=af)
    if verbose: print("GRAPPA Reconstruction done!")

    del data
    rec = rec.permute(0,3,1,2)

    rec = rec.numpy()
    
    if verbose: print("Fixing shape and IFFT..")
    rec = fixShapeAndIFFT(rec, scan_info)

    if performRSS:
        if verbose: print("Performing RSS..")
        rec = rss(rec, axis=0)
    
    if save_npy:
        if verbose: print("Saving reconstruction as Numpy array..")
        np.save(savepath, rec)
    
    if save_nii:
        if verbose: print("Saving reconstruction as Nifti..")
        save_nifti(rec, savepath)


if __name__ == "__main__":
    import sys
    Twix_GRAPPA_Recon(sys.argv[1], savepath=sys.argv[2] if len(sys.argv) > 2 else None)
