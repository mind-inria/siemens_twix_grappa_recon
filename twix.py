import mapvbvd
import numpy as np
import math
import torch
import scipy
import nibabel as nib

from grappa.grappaND import GRAPPA_Recon


class SiemensTwixReco:

    def __init__(self, filepath, **kwargs):

        self.twix = mapvbvd.mapVBVD(filepath)
        self.twix.image.flagRemoveOS = True
        self.twix.image.flagDoAverage = True
        self.twix.image.flagIgnoreSeg = True

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

        self.doNoiseDecorr = 'noise' in self.twix
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
            for s in range(self.NSli):
                self.cSli = s
                self.cSliSort = self.chronSlices[s]
                if self.cSliSort > self.NSli/self.multibandFactor:
                    continue

                self.cSliSort = (self.cSliSort + np.arange(self.multibandFactor) * self.NSli/self.multibandFactor).astype(int)

                for f in self.NFra:
                    self.cFra = self.NFra[f]
                    self.cRep = math.ceil(self.NFra[f]/self.NSet)
                    self.cSet = ((self.NFra[f] - 1) % self.NSet)
                    self._readSig()
                    self._performReco()
                    self._saveImg()


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

        self.sig = GRAPPA_Recon(self.sig, acs, af=self.af, **self.kwargs)
        self.sig = self.sig.permute(0,3,1,2)
        self.sig = self.sig.cpu().numpy()


    def _performReco(self):
        if self.doNoiseDecorr:
            self.sig = self._performNoiseDecorr(self.sig)

        self._performGrappa()
        self._fixShapeAndIFFT()


    def saveToNifTI(self, filepath):
        try:
            assert self.img is not None
            scan_img = nib.Nifti1Image(self.img.squeeze(), affine=np.eye(4))
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
