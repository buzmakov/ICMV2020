# %%*- coding: utf-8 -*-
# % matplotlib inline

# %%
from tomopy.misc.phantom import shepp2d, shepp3d
import matplotlib
import numpy as np
import h5py
import pylab as plt
from tqdm import tqdm
from tomo.recon.astra_utils import astra_recon_2d_parallel, astra_fp_2d_parallel
from cv2 import medianBlur
from scipy import interpolate as interp

matplotlib.rcParams.update({'font.size': 16})
# %%
data = np.squeeze(shepp2d(256)).astype('float32')
data /= data.max()
data = np.pad(data, data.shape[0] // 4, mode='constant')
angles = np.arange(0, 180, 1)

origin_sinogram = astra_fp_2d_parallel(data, angles)
rec = astra_recon_2d_parallel(origin_sinogram, angles)

# %%
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.imshow(data, cmap=plt.cm.gray)
plt.subplot(222)
plt.imshow(origin_sinogram, cmap=plt.cm.gray)
plt.axis('tight')
plt.subplot(223)
plt.imshow(rec, cmap=plt.cm.gray)
plt.subplot(224)
plt.imshow(data - rec, cmap=plt.cm.seismic)
plt.show()


# %%
def create_circle_mask(size):
    X, Y = np.meshgrid(np.arange(size), np.arange(size))
    X -= size // 2
    Y -= size // 2
    mask = (X ** 2 + Y ** 2) < (size // 2) ** 2 - 10
    return mask


def create_masked_sinogram(sino, mask, mode):
    res = sino.copy()
    if mode == 'constant':
        res[mask == False] = 0
    return res


def fix_radon(sino, mask=None):
    if mask is None:
        radon_inv = sino.sum(axis=-1)
    else:
        radon_inv = (sino * mask).sum(axis=-1) / np.mean(mask.astype(np.float32), axis=-1)
    fixed_sino = sino.T / radon_inv * np.mean(radon_inv)
    return fixed_sino.T


def interpolate_sino(image, mask):
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    data = image.copy()
    my_interp_func = interp.NearestNDInterpolator((x[mask], y[mask]), data[mask])
    z = my_interp_func(x, y)
    return z


# plt.figure(figsize=(15, 10))
# plt.imshow(interpolate_sino(sino, mask))
# plt.show()

def recon_with_mask(sinogram: np.ndarray, angles: np.ndarray, mask: np.ndarray, niters=5, method=[['FBP_CUDA']]):
    assert sinogram.shape == mask.shape
    circ_mask = create_circle_mask(sinogram.shape[1])
    t_sino = sinogram.copy() * mask
    # t_sino = interpolate_sino(t_sino, mask)
    rec = np.zeros_like(circ_mask)
    for i in tqdm(range(niters)):
        # t_sino = fix_radon(t_sino, mask)
        for j in range(i+1):
            rec = astra_recon_2d_parallel(t_sino, angles, method=method, data=None)
            # rec *= circ_mask
            # rec[rec < 0] /= 2
            t_sino = astra_fp_2d_parallel(rec, angles)
            t_sino[mask] = sinogram[mask]
            # t_sino[np.invert(mask)] *= np.random.normal(
            #     1.0, ((niters-i)/niters)*1e-2, np.sum(np.invert(mask)))
    return rec, t_sino


mask = np.ones_like(origin_sinogram, dtype=np.bool)
begin_stripe = mask.shape[1] // 4 + 13
# mask[:mask.shape[0] // 2, begin_stripe:begin_stripe + 10] = False
mask[:, begin_stripe:begin_stripe + 10] = False
sino = create_masked_sinogram(origin_sinogram, mask, 'constant')
mask_recon, res_sino = recon_with_mask(sino, angles, mask,
                                       niters=30,
                                       method=[['SART_CUDA', 50],
                                               ['SIRT_CUDA', 100],
                                               ])

# plt.figure(figsize=(15, 10))
# plt.subplot(231)
# plt.imshow(data, cmap=plt.cm.gray)
# plt.subplot(232)
# plt.imshow(sino, cmap=plt.cm.gray)
# plt.colorbar(orientation='horizontal')
# plt.subplot(233)
# plt.imshow(res_sino, cmap=plt.cm.gray)
# plt.colorbar(orientation='horizontal')
#
# plt.subplot(234)
# plt.imshow(mask_recon, cmap=plt.cm.gray)
# plt.subplot(235)
# plt.imshow(data-mask_recon, cmap=plt.cm.viridis)
# plt.subplot(236)
# plt.imshow(mask, cmap=plt.cm.gray)
# plt.show()


plt.figure(figsize=(15, 10))

plt.subplot(121)
plt.imshow(res_sino, cmap=plt.cm.gray)
plt.axis('tight')
plt.subplot(122)
plt.imshow(mask_recon[50:-50, 50:-50], vmin=0, vmax=1,
           cmap=plt.cm.gray)
plt.show()
