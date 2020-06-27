# %%*- coding: utf-8 -*-
# % matplotlib inline

import matplotlib
import numpy as np
import pylab as plt
from scipy import interpolate as interp
from tomopy.misc.phantom import shepp2d
from tqdm import tqdm

from tomo.recon.astra_utils import astra_recon_2d_parallel, astra_fp_2d_parallel

matplotlib.rcParams.update({'font.size': 16})

# %%
def create_circle_mask(size):
    X, Y = np.meshgrid(np.arange(size), np.arange(size))
    X -= size // 2
    Y -= size // 2
    mask = (X ** 2 + Y ** 2) < (size // 2 - 5) ** 2
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


def show_sino_rec(sino, recon):
    plt.figure(figsize=(15, 10))
    plt.subplot(121)
    plt.imshow(sino, cmap=plt.cm.gray)
    plt.axis('tight')
    plt.subplot(122)
    plt.imshow(recon[50:-50, 50:-50], vmin=0, vmax=1,
               cmap=plt.cm.gray)
    plt.show()

default_method = [['SART_CUDA', 50],
                 ['SIRT_CUDA', 100]]

def recon_with_mask(sinogram: np.ndarray, angles: np.ndarray, mask: np.ndarray,
                    niters = 100,
                    method = default_method):
    assert sinogram.shape == mask.shape
    t_sino = sinogram.copy() * mask
    rec = np.zeros((sinogram.shape[1], sinogram.shape[1]), dtype='float32')
    for i in tqdm(range(niters)):
        rec = astra_recon_2d_parallel(t_sino, angles, method=method, data=None)
        t_sino = astra_fp_2d_parallel(rec, angles)
        t_sino[mask] = sinogram[mask]
        if i % 10 == 0:
            show_sino_rec(t_sino, rec)
            # print(kk)
    return rec, t_sino


def generate_sinogram(data_size, angles):
    data = np.squeeze(shepp2d(data_size)).astype('float32')
    data /= data.max()
    data = np.pad(data, data.shape[0] // 4, mode='constant')
    origin_sinogram = astra_fp_2d_parallel(data, angles)
    return origin_sinogram, angles, data


def test_case_1():
    origin_sinogram, angles, data = generate_sinogram(128, np.arange(0, 180, 1))

    rec = astra_recon_2d_parallel(origin_sinogram, angles)

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

    mask = np.ones_like(origin_sinogram, dtype=np.bool)
    begin_stripe = mask.shape[1] // 4 + 13
    mask[:, begin_stripe:begin_stripe + 10] = False
    sino = create_masked_sinogram(origin_sinogram, mask, 'constant')

    mask_recon, res_sino = recon_with_mask(sino, angles, mask)



    show_sino_rec(res_sino, mask_recon)


def test_case_2():
    origin_sinogram, angles, data = generate_sinogram(128, np.arange(0, 180, 1))

    rec = astra_recon_2d_parallel(origin_sinogram, angles)

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

    mask = np.ones_like(origin_sinogram, dtype=np.bool)
    begin_stripe = mask.shape[1] // 4 + 13
    mask[:mask.shape[0] // 2, begin_stripe:begin_stripe + 10] = False
    sino = create_masked_sinogram(origin_sinogram, mask, 'constant')

    mask_recon, res_sino = recon_with_mask(sino, angles, mask)

    show_sino_rec(res_sino, mask_recon)


def test_case_3():
    origin_sinogram, angles, data = generate_sinogram(128, np.arange(0, 360, 1))

    rec = astra_recon_2d_parallel(origin_sinogram, angles)

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

    mask = np.ones_like(origin_sinogram, dtype=np.bool)
    begin_stripe = mask.shape[1] // 4 + 13
    mask[:, :mask.shape[1] //2 - 10] = False
    sino = create_masked_sinogram(origin_sinogram, mask, 'constant')

    mask_recon, res_sino = recon_with_mask(sino, angles, mask)

    show_sino_rec(res_sino, mask_recon)

test_case_3()