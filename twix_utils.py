import numpy as np


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
        sig_r[nc] = np.fft.fftshift(np.fft.ifft(np.fft.ifft(np.fft.ifft(np.fft.fftshift(sig_r[nc], axes=(0,1,2)), axis=0), axis=1), axis=2), axes=(0,1,2))

    return sig_r
