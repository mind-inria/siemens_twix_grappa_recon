import os
import torch
import numpy as np

from grappaND import GRAPPA_Recon
from twix_reader import read_twix_datafile
from twix_utils import fixShapeAndIFFT
from utils import rss


def _performNoiseDecorr(data, noise):
    R = torch.cov(noise)
    R = R/np.mean(np.abs(np.diag(R)))
    R[torch.eye(R.size(0), dtype=torch.bool)] = np.abs(np.diag(R))
    R_inv = torch.linalg.inv(R)
    R_inv_sqrt = torch.from_numpy(np.real(np.linalg.sqrtm(R_inv.numpy())))
    D = R_inv_sqrt.T

    data_sz = data.shape
    data_decorr = D @ data.reshape(data.shape[0], -1)

    return data_decorr.reshape(data_sz)


def Twix_GRAPPA_Recon(filepath, savepath=None, performRSS=True):

    data, scan_info = read_twix_datafile(filepath)
    data['image'] = _performNoiseDecorr(data['image'], data['noise'])

    af = []
    if ('sPat', 'lAccelFact3D') in scan_info.hdr.MeasYaps:
        af.append(int(scan_info.hdr.MeasYaps[('sPat', 'lAccelFact3D')]))
    if ('sPat', 'lAccelFactPE') in scan_info.hdr.MeasYaps:
        af.append(int(scan_info.hdr.MeasYaps[('sPat', 'lAccelFactPE')]))

    rec = GRAPPA_Recon(data['image'], data['refscan'], af=af)

    del data

    rec = rec.numpy()
    rec = fixShapeAndIFFT(rec, scan_info)

    if performRSS:
        rec = rss(rec, axis=0)
    
    if not savepath:
        base, ext = os.path.splitext(filepath)
        savepath = base + "_GRAPPArecon" + ext
    
    np.save(savepath, rec)
