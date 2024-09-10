import mapvbvd
import numpy as np
import math
import torch
import scipy
import nibabel as nib
import pydicom
import logging
import cv2
import os
import json
import time
import sys

from timeit import default_timer as timer

from grappa.grappaND import GRAPPA_Recon
from tqdm import tqdm

from twix_utils import range_normalize


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SiemensTwixReco:
    def __init__(self, filepath, **kwargs):
        self.filepath = filepath
        self.twix = mapvbvd.mapVBVD(self.filepath, quiet=True)

        if isinstance(self.twix, list):
            self.twix = self.twix[-1]
        
        self.doNoiseDecorr = 'noise' in self.twix
        self.doPhaseCorr = 'phasecor' in self.twix
        self.doRefPhaseCorr = 'refscanPC' in self.twix
        self.doRampRegrid = 'alRegridMode' in self.twix.hdr.Meas and int(self.twix.hdr.Meas.alRegridMode[0]) > 1
        self.doPreRecoOSRemoval = True

        fields = self.twix.keys()

        for name in fields:
            if name != 'hdr':
                self.twix[name].flagRemoveOS = name not in ['noise', 'vop']
                self.twix[name].flagDoAverage = True
                self.twix[name].flagSkipToFirstLine = name not in ['image']
                self.twix[name].flagRampSampRegrid = name not in ['noise', 'vop'] and self.doRampRegrid
                self.twix[name].flagIgnoreSeg = not self.doPhaseCorr
        
        self.NCha = int(self.twix.image.NCha)
        self.NCol = int(self.twix.image.NCol) // 2
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

        self.sig = None
        self.grappa_kernel = None
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
        self.sig = self.twix.image[:,:,:,:,self.cSli,0,0,self.cEco,self.cRep,self.cSet,:].swapaxes(0,1)
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
        
    def _performPhaseCorr(self, sig, pc_obj):
        if self.doPhaseCorr:

            pc = pc_obj[:,:,:,:,self.cSli,:,0,min(pc_obj.NEco, self.cEco),
                        min(pc_obj.NRep, self.cRep),min(pc_obj.NSet, self.cSet),:].swapaxes(0,1)[:,:,:,:,0,0,0,0,0,0,:,0,0,0,0,0]

            if self.doNoiseDecorr:
                pc = self._performNoiseDecorr(pc)

            pc = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(pc, axes=1), axis=1, norm='ortho'), axes=1)

            pc_method = 'autocorr'

            match pc_method:
                # case "autocorracrossseg":
                #     x = np.arange(-np.floor(self.NCol / 2), np.ceil(self.NCol / 2))

                #     dphi1 = np.conj(np.multiply(pc[:, 1:, 0, 0, 1:], np.conj(pc[:, 1:, 0, 0, 0]))) * np.conj(np.multiply(pc[:, :-1, 0, 0, 1:], np.conj(pc[:, :-1, 0, 0, 0])))
                #     dphi1 = np.sum((self.NCol - 1) * dphi1, axis=1)
                #     dphi1 = np.sum(dphi1, axis=0)
                #     dphi1 = np.angle(dphi1)

                #     dphi0 = np.multiply(pc[:, 1:-1, 0, 0, 0], np.conj(pc[:, 1:-1, 0, 0, 1:]))
                #     dphi0 = np.multiply(dphi0, np.exp(1j * np.multiply(x[1:-1], dphi1)))
                #     dphi0 = np.sum(np.multiply(self.NCol - 2, dphi0), axis=1)
                #     dphi0 = np.sum(dphi0, axis=0)
                #     dphi0 = -np.angle(dphi0)

                #     pc = -np.add(dphi0, np.multiply(dphi1, x)) / 2
                #     pc[:, :, :, :, pc.shape[4] + 1] = -pc
                #     pc = np.exp(1j * pc)

                case "autocorr":
                    x = np.linspace(-0.5, 0.5, self.NCol)

                    slope = self.NCol * np.angle(np.sum(np.sum(np.conj(pc[:, :-1, 0, 0, :]) * pc[:, 1:, 0, 0, :], axis=1), axis=0))
                    intercept = 0

                    pc = np.outer(x, slope) + intercept

                    pc = np.exp(1j * pc)
                    pc = pc[None, :, None, None, :]
                

            sig = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(sig, axes=1), axis=1, norm='ortho'), axes=1)
            sig = pc.conj() * sig
            sig = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(sig, axes=1), axis=1, norm='ortho'), axes=1)

            return sig
    
    def _performRampRegrid(self, sig=None):
        if self.doRampRegrid:
            if sig is None:
                self.sig = self.sig.sum(-1)
            else:
                sig = sig.sum(-1)
                return sig

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
                self.acs = None

                for f in self.NFra:
                    self.cFra = self.NFra[f]
                    self.cRep = math.ceil((self.cFra+1)/self.NSet) - 1
                    self.cSet = (self.cFra % self.NSet)

                    self._readSig()
                    self._performReco()
                    self._saveImg()

                    torch.cuda.empty_cache()

    def _performGrappa(self):
        first_read_acs = self.acs is None

        if first_read_acs:
            self.acs = self.twix.refscan[:,:,:,:,self.cSli,0,0,0,0,0,:].swapaxes(0,1)
            self.acs = self.acs.squeeze()
        
        if first_read_acs and self.doNoiseDecorr:
            self.acs = self._performNoiseDecorr(self.acs)

        if first_read_acs and self.doRefPhaseCorr:
            self.acs = self._performPhaseCorr(self.acs, self.twix.refscanPC)
        
        if first_read_acs and self.doRampRegrid:
            self.acs = self._performRampRegrid(self.acs)
        
        if len(self.sig.shape) == 3:
            self.sig = self.sig[:, :, :, None]

        if len(self.acs.shape) == 3:
            self.acs = self.acs[:, :, :, None]
        
        self.acs = torch.from_numpy(self.acs) if first_read_acs and type(self.acs) is not torch.Tensor else self.acs
        self.sig = torch.from_numpy(self.sig) if type(self.sig) is not torch.Tensor else self.sig

        self.sig = self.sig.permute(0,2,3,1)
        self.acs = self.acs.permute(0,2,3,1) if first_read_acs else self.acs

        self.sig, self.grappa_kernel = GRAPPA_Recon(self.sig, self.acs, af=self.af, grappa_kernel=self.grappa_kernel, mask=mask, **self.kwargs)
        
        self.sig = self.sig.permute(0,3,1,2)
        self.sig = self.sig.cpu().numpy()

    def _performReco(self):
        if self.doNoiseDecorr:
            self.sig = self._performNoiseDecorr(self.sig)

        if self.doPhaseCorr:
            self.sig = self._performPhaseCorr(self.sig, self.twix.phasecor)

        self._performRampRegrid()
        self._performGrappa()
        self._fixShapeAndIFFT()

    def saveToNifTI(self, filepath, to_dicom_range=False):
        try:
            assert self.img is not None
            self.img = self.img.squeeze()
            #self.img = np.flip(self.img, 0)
            #self.img = self.img.swapaxes(0,2).swapaxes(1,2)
            if len(self.NFra) == 1:
                if to_dicom_range:
                    self.img = np.int16(range_normalize(self.img, 0, 4095))
                scan_img = nib.Nifti1Image(self.img, affine=np.eye(4)) # get_siemens_twix_rotation_matrix(self.filepath))
                nib.save(scan_img, filepath)
            else:
                for f in self.NFra:
                    img = self.img[...,f]
                    if to_dicom_range:
                        img = np.int16(range_normalize(img, 0, 4095))
                    scan_img = nib.Nifti1Image(img, affine=np.eye(4)) # affine=get_siemens_twix_rotation_matrix(self.filepath))
                    nib.save(scan_img, f"{filepath}_{f}")

        except AssertionError as e:
            raise Exception("You need to call performReco() first to populate the 'self.img' variable.") from e

    def saveToNpy(self, filepath):
        try:
            assert self.img is not None
            np.save(filepath, self.img.squeeze())
        except AssertionError as e:
            raise Exception("You need to call performReco() first to populate the 'self.img' variable.") from e

    # def saveToDICOM(self, filepath):
    #     img = np.int16(range_normalize(self.img, 0, 4095))
    #     uid = pydicom.uid.generate_uid()
    #     for s in range(self.img.shape[2]):
    #         ds = pydicom.Dataset()
    #         ds.PatientID = '123456'
    #         ds.Modality = 'MR'
    #         ds.StudyDate = '20240908'
    #         ds.SeriesDate = uid
    #         ds.InstanceNumber = s+1
    #         ds.Rows = img.shape[0]
    #         ds.Columns = img.shape[1]
    #         ds.PixelSpacing = [0.8, 0.8]
    #         ds.BitsAllocated = 16
    #         ds.BitsStored = 12
    #         ds.PhotometricInterpretation = "MONOCHROME2"
    #         ds.PixelRepresentation = 0  # Unsigned integer
    #         ds.SamplesPerPixel = 1

    #         # Set Pixel Data tag
    #         ds.PixelData = img[...,s,0].tostring()
    #         ds.is_little_endian = True
    #         ds.is_implicit_VR = True

    #         # Save the DICOM file
    #         ds.save_as(os.path.join(filepath, f"{uid}-{s}.dcm"))

    def _saveImg(self):
        if self.img is None:
            sz = [1,1,1,1,1]
            sz[:len(self.sig.shape)] = self.sig.shape
            sz[0] = 1

            self.img = np.zeros(tuple(sz[:4] + [self.NSli, len(self.NFra), self.NEco]), dtype=self.sig.dtype)

        self.img[0,..., self.cSliSort, self.cFra, self.cEco] = np.sqrt(np.sum(np.abs(np.flip(self.sig, 1))**2, axis=0))


if __name__ == "__main__":
    filename = sys.argv[1]
    scan = SiemensTwixReco(filename, kernel_size=(4,4,5), cuda=True, cuda_mode="application", lambda_=1e-4, batch_size=10, verbose=True)
    scan.runReco()
    scan.saveToNifTI(sys.argv[2])
