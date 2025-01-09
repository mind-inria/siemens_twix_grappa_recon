import os

import numpy as np
import torch
import scipy


def fixShapeAndIFFT(sig, scan):
    sz = sig.shape
    trg_sz = [sig.shape[0], int(scan.hdr.Meas.iRoFTLength)//2, int(scan.hdr.Meas.iPEFTLength), int(scan.hdr.Meas.i3DFTLength)]
    diff_sz = np.array(trg_sz) - np.array(sz)
    shift = np.floor(diff_sz / 2).astype(int)

    ixin1 = np.maximum(shift, 0)
    ixin2 = np.maximum(shift, 0) + np.minimum(sz, trg_sz)
    ixout1 = np.abs(np.minimum(shift, 0))
    ixout2 = np.abs(np.minimum(shift, 0)) + np.minimum(sz, trg_sz)

    sig_r = np.zeros(trg_sz, dtype=sig.dtype)

    sig_r[:, ixin1[1]:ixin2[1], ixin1[2]:ixin2[2], ixin1[3]:ixin2[3]] = sig[:, ixout1[1]:ixout2[1], ixout1[2]:ixout2[2], ixout1[3]:ixout2[3]]

    for nc in range(sig_r.shape[0]):
        sig_tmp = torch.from_numpy(sig_r[nc]).cuda()
        sig_tmp = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(sig_tmp, dim=(0,1,2)), dim=(0,1,2), norm='ortho'), dim=(0,1,2))
        sig_r[nc] = sig_tmp.cpu().numpy()

    return sig_r
    

def performNoiseDecorr(data, noise):
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


def range_normalize(src, alpha=0, beta=1):
    """
    Normalizes an array to a specified range.

    Parameters:
    - src: input array.
    - alpha: lower range boundary.
    - beta: upper range boundary.

    Returns:
    - normalized src
    """
    min_val = np.min(src)
    max_val = np.max(src)
    return (src - min_val) / (max_val - min_val) * (beta - alpha) + alpha


def ifftn(sig, dims):
    return np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(sig, axes=dims), axes=dims), axes=dims)


def siemens_quat_to_rot_mat(quat):
    """
    Calculate the rotation matrix from Siemens Twix quaternion.
    """
    a = quat[1]
    b = quat[2]
    c = quat[3]
    d = quat[0]

    R = np.zeros((4, 4))
    
    R[0,1] = 1.0 - 2.0 * (b * b + c * c)
    R[0,0] = 2.0 * (a * b - c * d)
    R[0,2] = 2.0 * (a * c + b * d)

    R[1,1] = 2.0 * (a * b + c * d)
    R[1,0] = 1.0 - 2.0 * (a * a + c * c)
    R[1,2] = 2.0 * (b * c - a * d)

    R[2,1] = 2.0 * (a * c - b * d)
    R[2,0] = 2.0 * (b * c + a * d)
    R[2,2] = 1.0 - 2.0 * (a * a + b * b)

    if (np.linalg.det(R[:3, :3]) < 0):
        R[2] = -R[2]
        
    R[-1,-1] = 1

    return R


def get_filename(filepath, twix_obj, frame=None):
    filename = (
        f"{os.path.join(os.path.abspath(filepath), os.path.splitext(os.path.basename(twix_obj.filepath))[0])}"
        f"{f'__frame_{frame}' if frame else ''}__gGRAPPAReco__ksz_{'_'.join(map(str, twix_obj.kwargs.get('kernel_size', ['4', '4', '5'])))}"
        f"__l__{str(twix_obj.kwargs.get('lambda_', str(1e-4))).replace('.','_')}"
    )
    return filename
