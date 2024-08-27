import mapvbvd
import numpy as np
import math
import torch
import scipy
import nibabel as nib
import cv2
import os
import json
import time

from timeit import default_timer as timer

from grappa.grappaND import GRAPPA_Recon
from tqdm import tqdm

from twix_utils import minmax_normalize


def siemens_quat_to_rot_mat(q):
    """
    Calculate the rotation matrix from Siemens Twix quaternion.
    """
    ds = 2.0 / np.sum(q**2)
    dxs = q[1] * ds
    dys = q[2] * ds
    dzs = q[3] * ds
    dwx = q[0] * dxs
    dwy = q[0] * dys
    dwz = q[0] * dzs
    dxx = q[1] * dxs
    dxy = q[1] * dys
    dxz = q[1] * dzs
    dyy = q[2] * dys
    dyz = q[2] * dzs
    dzz = q[3] * dzs
    
    R = np.zeros((4, 4))
    R[0, 0] = 1.0 - (dyy + dzz)
    R[0, 1] = dxy + dwz
    R[0, 2] = dxz - dwy
    R[1, 0] = dxy - dwz
    R[1, 1] = 1.0 - (dxx + dzz)
    R[1, 2] = dyz + dwx
    R[2, 0] = dxz + dwy
    R[2, 1] = dyz - dwx
    R[2, 2] = 1.0 - (dxx + dyy)
    
    R[-1,-1] = 1
        
    return R


def siemens_quat_to_rot_mat2(quat):
    """
    Calculate the rotation matrix from Siemens Twix quaternion.
    """
    a = quat[0]
    b = quat[1]
    c = quat[2]
    d = quat[3]

    R = np.zeros((4, 4))
    
    R[0,0] = 1.0 - 2.0 * (b * b + c * c)
    R[0,1] = 2.0 * (a * b - c * d)
    R[0,2] = 2.0 * (a * c + b * d)

    R[1,0] = 2.0 * (a * b + c * d)
    R[1,1] = 1.0 - 2.0 * (a * a + c * c)
    R[1,2] = 2.0 * (b * c - a * d)

    R[2,0] = 2.0 * (a * c - b * d)
    R[2,1] = 2.0 * (b * c + a * d)
    R[2,2] = 1.0 - 2.0 * (a * a + b * b)
    R[-1,-1] = 1

    return R


def get_siemens_twix_rotation_matrix(filepath):
    """
    Extract the orientation matrix from Siemens Twix (twixtools) scan object.
    """
    from twixtools import read_twix
    twix_obj = read_twix(filepath)
    if type(twix_obj) is list:
        twix_obj = twix_obj[-1]
    mdb = twix_obj['mdb']
    mdh = mdb[0].mdh
    quat = mdh.SliceData.Quaternion
    return siemens_quat_to_rot_mat(quat)



class SiemensTwixReco:

    def __init__(self, filepath, **kwargs):
        self.filepath = filepath
        self.twix = mapvbvd.mapVBVD(self.filepath)

        if isinstance(self.twix, list):
            self.twix = self.twix[-1]
        
        self.doNoiseDecorr = 'noise' in self.twix
        self.doPhaseCorr = 'phasecor' in self.twix
        self.doRefPhaseCorr = 'refscanPC' in self.twix
        self.doRampRegrid = 'alRegridMode' in int(self.twix.hdr.Meas and self.twix.hdr.Meas.alRegridMode[0]) > 1
        self.doPreRecoOSRemoval = True

        fields = self.twix.keys()

        for name in fields:
            self.twix[name].flagRampSampRegrid = name not in ['noise', 'vop'] and self.doRampRegrid
        
        self.twix.image.flagRemoveOS = True
        self.twix.image.flagDoAverage = True
        self.twix.image.flagIgnoreSeg = True
        self.twix.image.flag

        self.twix.refscan.flagRemoveOS = True
        self.twix.refscan.flagDoAverage = True
        self.twix.refscan.flagIgnoreSeg = True
        
        self.NCha = int(self.twix.image.NCha)
        self.NCol = int(self.twix.image.NCol)
        self.NLin = int(self.twix.image.NLin)
        self.NPar = int(self.twix.image.NPar)
        self.NSli = int(self.twix.image.NSli)
        self.NEco = int(self.twix.image.NEco)
        self.NRep = int(self.twix.image.NRep)
        self.NSet = int(self.twix.image.NSet)
        self.NFra = np.arange(self.NSet * self.NRep)
        self.centCol = int(self.twix.image.centerCol[0])
        self.centLin = int(self.twix.image.centerLin[0])
        self.centPar = int(self.twix.image.centerPar[0])
        self.recon_times = {}

        self.sig = None
        self.img = None
        self.multibandFactor = 1

        if 'chronSliceIndices' in self.twix.hdr.Meas:
            self.chronSlices = np.fromstring(self.twix.hdr.Meas.chronSliceIndices, dtype=int, sep=' ')[:self.NSli]# +1
        else:
            self.chronSlices = np.arange(self.NSli)
        
        self.af = [int(self.twix.hdr.MeasYaps[('sPat', 'lAccelFactPE')]), int(self.twix.hdr.MeasYaps[('sPat', 'lAccelFact3D')])]

        if self.NPar == 1:
            self.af[1] = 1


        self.kwargs = kwargs

    def _readSig(self):
        self.sig = self.twix.image[:,:,:,:,self.cSli,0,0,self.cEco,self.cRep,self.cSet].swapaxes(0,1)
        self.sig = self.sig.squeeze()
        self.sig = torch.from_numpy(self.sig)

    def _calcNoiseDecorrMatrix(self):
        if 'noise' in self.twix:
            noise = self.twix.noise[''].swapaxes(0,1).squeeze().T
            R = np.cov(noise, rowvar=False)
            mean_abs_diag = np.mean(np.abs(np.diag(R)))
            R = R / mean_abs_diag
            np.fill_diagonal(R, np.abs(np.diag(R)))
            R_inv = np.linalg.inv(R)
            R_inv_sqrt = scipy.linalg.sqrtm(R_inv).astype(np.complex64)

            self.D = torch.from_numpy(R_inv_sqrt)
        else:
            self.D = np.eye(self.NCha)
        
    def _calcPhaseCorr(self. sig, pc_obj):
        if self.doPhaseCorr:
            self.twix.phasecor.flagRemoveOS = True
            self.twix.phasecor.flagIgnoreSeg = False
            self.twix.phasecor.flagDoAverage = True
            self.twix.phasecor.flagRampSampRegird = True
            self.twix.phasecor.flagSkipToFirstLine = True

            pc = pc_obj[:,:,:,:,:].squeeze().swapaxes(0,1)

            if self.doNoiseDecorr:
                pc = self._performNoiseDecorr(pc)

            pc = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(pc, axes=1), axis=1), axes=1)

            #wi = np.sqrt(np.sum(np.abs(pc[:,:,:,:,:])**2, axis=4))

            pc_method = 'autocorracrossseg'

            match pc_method:
                # case "autocorracrossseg":
                #     ncol = self.NCol
                
                #     # ifft col dim.
                #     pc = ifftn(self.sig, [-1])
                    
                #     # calculate phase slope from autocorrelation (for both readout polarities separately - each in its own dim)
                #     slope = np.angle((np.conj(pc[1:]) * pc[:-1]).sum(0, keepdims=True).sum(1, keepdims=True))
                #     x = np.arange(ncol) - ncol//2
                    
                #     return np.exp(1j * slope * x)
                case "auto":
                    x = np.linspace(-0.5, 0.5, self.NCol)
                    slope = self.NCol * np.angle(np.sum(np.sum(np.conj(pc[:, : -1, 0, :]) * pc[:, 1 :, 0, :], axis=1), axis=0))

                    intercept = 0
                    # Create the linear phase
                    pc = np.add.outer(slope, x) + intercept

                    # Apply the phase
                    pc = np.exp(1j * pc)
            
            sig = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(sig, axes=1), axis=1), axes=1)
            sig = pc.conj() * sig
            sig = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(pc, axes=1), axis=1), axes=1)

            return sig

    def _performPhaseCorr(self):
        pass

    def _performNoiseDecorr(self, data):
        data_sz = data.shape
        data = self.D @ data.reshape(data.shape[0], -1)
        data = data.reshape(data_sz)
        return data

    def _fixShapeAndIFFT(self):
        sz = self.sig.shape
        trg_sz = [self.sig.shape[0], int(self.twix.hdr.Meas.iRoFTLength)//2, int(self.twix.hdr.Meas.iPEFTLength), int(self.twix.hdr.Meas.i3DFTLength)]
        diff_sz = np.array(trg_sz) - np.array(sz)
        shift = np.floor(diff_sz / 2).astype(int)

        ixin1 = np.maximum(shift, 0)
        ixin2 = np.maximum(shift, 0) + np.minimum(sz, trg_sz)
        ixout1 = np.abs(np.minimum(shift, 0))
        ixout2 = np.abs(np.minimum(shift, 0)) + np.minimum(sz, trg_sz)

        sig_r = np.zeros(trg_sz, dtype=self.sig.dtype)

        sig_r[:, ixin1[1]:ixin2[1], ixin1[2]:ixin2[2], ixin1[3]:ixin2[3]] = self.sig[:, ixout1[1]:ixout2[1], ixout1[2]:ixout2[2], ixout1[3]:ixout2[3]]

        for nc in range(sig_r.shape[0]):
            sig_tmp = torch.from_numpy(sig_r[nc]).cuda()
            sig_tmp = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(sig_tmp, dim=(0,1,2)), dim=(0,1,2), norm='ortho'), dim=(0,1,2))
            sig_r[nc] = sig_tmp.cpu().numpy()
        
        self.sig = sig_r
    
    def runReco(self):

        if self.doNoiseDecorr:
            self._calcNoiseDecorrMatrix()

        for e in range(self.NEco):
            self.cEco = e
            for s in tqdm(range(self.NSli)):
                self.cSli = s
                self.cSliSort = self.chronSlices[s]
                if self.cSliSort > self.NSli/self.multibandFactor:
                    continue

                self.cSliSort = (self.cSliSort + np.arange(self.multibandFactor) * self.NSli/self.multibandFactor).astype(int)

                for f in self.NFra:
                    self.cFra = self.NFra[f]
                    self.cRep = math.ceil(self.NFra[f]/self.NSet)
                    self.cSet = ((self.NFra[f] - 1) % self.NSet)
                    before = timer()
                    self._readSig()
                    self.recon_times[f'after_read+{s}'] = timer() - before
                    bff = timer()
                    self._performReco()
                    self.recon_times[f'after_perform reco+{s}'] = timer()-bff
                    bff2=timer()
                    self._saveImg()
                    self.recon_times[f'saveImg+{s}'] = timer() - bff2
                    self.recon_times[f'read+reco+save+{s}'] = timer() - before

                    torch.cuda.empty_cache()

    def _performGrappa(self):
        acs = self.twix.refscan[:,:,:,:,self.cSli,0,0,0,0,0,].swapaxes(0,1)
        acs = acs.squeeze()
        if self.doNoiseDecorr: acs = self._performNoiseDecorr(acs)
        else: acs = torch.from_numpy(acs)
        if len(self.sig.shape) == 3:
            self.sig = self.sig[:, :, :, None]
        if len(acs.shape) == 3:
            acs = acs[:, :, :, None]

        self.sig = self.sig.permute(0,2,3,1)
        acs = acs.permute(0,2,3,1)
        before_recon = timer()
        self.sig = GRAPPA_Recon(self.sig, acs, af=self.af, **self.kwargs)
        if 'GRAPPA_recon_time' not in self.recon_times.keys():
            self.recon_times['GRAPPA_recon_time'] = timer() - before_recon
        self.sig = self.sig.permute(0,3,1,2)

        self.sig = self.sig.cpu().numpy()

    def _performReco(self):
        if self.doNoiseDecorr:
            self.sig = self._performNoiseDecorr(self.sig)

        self._performGrappa()
        bef = timer()
        self._fixShapeAndIFFT()
        if 'IFFTANDSHAPE' not in self.recon_times.keys():
            self.recon_times['IFFTANDSHAPE'] = timer() - bef

    def saveToNifTI(self, filepath, to_dicom_range=False):
        try:
            assert self.img is not None
            #scan_img = nib.Nifti1Image(self.img.squeeze(), affine=get_siemens_twix_rotation_matrix(self.filepath))
            self.img = self.img.real.squeeze()
            #self.img = self.img.swapaxes(0,1)
            #self.img = np.int16(cv2.normalize(self.img, None, 0, 4095, cv2.NORM_MINMAX))
            if to_dicom_range:
                self.img = np.int16(minmax_normalize(self.img, 0, 4095))
            scan_img = nib.Nifti1Image(self.img, affine=np.eye(4))
            nib.save(scan_img, filepath)
        except AssertionError as e:
            raise Exception("You need to call performReco() first to populate the 'self.img' variable.") from e

    def saveToNpy(self, filepath):
        try:
            assert self.img is not None

            np.save(filepath, self.img.squeeze())
        except AssertionError as e:
            raise Exception("You need to call performReco() first to populate the 'self.img' variable.") from e

    def _saveImg(self):
        if self.img is None:
            sz = [1,1,1,1,1]
            sz[:len(self.sig.shape)] = self.sig.shape
            sz[0] = 1

            self.img = np.zeros(tuple(sz[:4] + [self.NSli, len(self.NFra), self.NEco]), dtype=self.sig.dtype)

        self.img[0,..., self.cSliSort, self.cFra, self.cEco] = np.sqrt(np.sum(np.abs(self.sig)**2, axis=0))
