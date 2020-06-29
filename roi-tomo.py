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
    return mask.astype('float32')


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

def monitor_recon(i, rec, t_sino):
    plt.figure(figsize=(12, 10))
    plt.subplot(121)
    plt.imshow(t_sino, cmap=plt.cm.gray)
    plt.axis('tight')
    plt.subplot(122)
    plt.imshow(rec, vmin=0, vmax=1, cmap=plt.cm.gray)
    plt.title(i)
    plt.show()

def recon_with_mask(sinogram: np.ndarray, angles: np.ndarray, mask: np.ndarray,
                    niters=300,
                    method= [['FBP_CUDA']],
                    interpolation=False,
                    monitoring_iteration=None):
    assert sinogram.shape == mask.shape

    circle_mask = create_circle_mask(sinogram.shape[1], )
    t_sino = sinogram.copy() * mask
    if interpolation:
        t_sino = interpolate_sino(t_sino, mask)
    rec = np.zeros((sinogram.shape[1], sinogram.shape[1]), dtype='float32')
    k0 = np.sum(t_sino[mask])
    for i in tqdm(range(niters)):
        rec = astra_recon_2d_parallel(t_sino, angles, method=method, data=None)
        rec *= circle_mask
        t_sino = astra_fp_2d_parallel(rec, angles)
        t_sino = t_sino / np.sum(t_sino[mask]) * k0  # FBP normalization fix
        t_sino[mask] = sinogram[mask]
        if (monitoring_iteration is not None) and (i % monitoring_iteration == 0):
            monitor_recon(i, rec, t_sino)
    return rec, t_sino


def generate_sinogram(data_size, angles):
    data = np.squeeze(shepp2d(data_size)).astype('float32')
    data /= data.max()
    data = np.pad(data, data.shape[0] // 4, mode='constant')
    origin_sinogram = astra_fp_2d_parallel(data, angles)
    return origin_sinogram, angles, data


def do_test(sinogram, data, angles, mask, nitres=500, monitoring_iteration=None):
    rec = astra_recon_2d_parallel(sinogram, angles)

    # plt.figure(figsize=(12, 12))
    # plt.subplot(221)
    # plt.imshow(data, cmap=plt.cm.gray)
    # plt.subplot(222)
    # plt.imshow(sinogram, cmap=plt.cm.gray)
    # plt.axis('tight')
    # plt.subplot(223)
    # plt.imshow(rec, cmap=plt.cm.gray)
    # plt.subplot(224)
    # plt.imshow(data - rec, cmap=plt.cm.seismic)
    # plt.show()

    recon_my, res_sino = recon_with_mask(sinogram, angles, mask,
                                         niters=nitres,
                                         monitoring_iteration=monitoring_iteration)

    rec_corrupted = astra_recon_2d_parallel(sinogram * mask, angles)

    rec_bad_reg = astra_recon_2d_parallel(1-mask, angles, method=[["BP_CUDA"]])

    plt.figure(figsize=(15, 10))

    plt.subplot(231)
    plt.imshow(sinogram , cmap=plt.cm.gray)
    plt.imshow(np.ma.masked_where(mask == 1, mask), cmap=plt.cm.hsv, alpha=0.3)
    plt.axis('tight')
    plt.title('Masked sinogram')

    plt.subplot(234)
    plt.imshow(res_sino, cmap=plt.cm.gray)
    plt.axis('tight')
    plt.title('Reconstructed sinogram')

    plt.subplot(232)
    plt.imshow(rec_bad_reg, cmap=plt.cm.gray)
    plt.title('Untrusted region')

    plt.subplot(233)
    plt.imshow(rec, vmin=0, vmax=1,
               cmap=plt.cm.gray)
    plt.title('Ideal reconstruction')

    plt.subplot(235)
    plt.imshow(rec_corrupted, vmin=0, vmax=1,
               cmap=plt.cm.gray)
    # plt.imshow(np.ma.masked_where(rec_bad_reg == 0, rec_bad_reg), cmap=plt.cm.Reds, alpha=0.3)
    plt.title('Mask recon')

    plt.subplot(236)
    plt.imshow(recon_my, vmin=0, vmax=1,
               cmap=plt.cm.gray)
    plt.title('My recon')

    plt.show()


def test_case_1(): # column
    angles = np.arange(0, 180, 0.2)
    data_size = 128
    origin_sinogram, angles, data = generate_sinogram(data_size, angles)

    mask = np.ones_like(origin_sinogram, dtype=np.bool)
    begin_stripe = mask.shape[1] // 4 + 13
    mask[:, begin_stripe:begin_stripe + 10] = False

    do_test(origin_sinogram, data, angles, mask)

def test_case_2(): # center
    angles = np.arange(0, 180, 0.2)
    data_size = 128
    origin_sinogram, angles, data = generate_sinogram(data_size, angles)

    mask = np.ones_like(origin_sinogram, dtype=np.bool)
    begin_stripe = mask.shape[1] // 2 -5
    mask[:, begin_stripe:begin_stripe + 10] = False

    do_test(origin_sinogram, data, angles, mask)


def test_case_3(): # half of column
    angles = np.arange(0, 180, 0.2)
    data_size = 128
    origin_sinogram, angles, data = generate_sinogram(data_size, angles)

    mask = np.ones_like(origin_sinogram, dtype=np.bool)
    begin_stripe = mask.shape[1] // 4 + 13
    mask[:mask.shape[0] // 2, begin_stripe:begin_stripe + 10] = False

    do_test(origin_sinogram, data, angles, mask)


def test_case_4():  # alfa aquisition
    angles = np.arange(0, 360, 0.2)
    data_size = 128

    origin_sinogram, angles, data = generate_sinogram(data_size, angles)

    mask = np.ones_like(origin_sinogram, dtype=np.bool)
    mask[:, :mask.shape[1] // 2 - 10] = False

    do_test(origin_sinogram, data, angles, mask)


def test_case_5():
    angles = np.arange(0, 180, 0.2)
    data_size = 128

    origin_sinogram, angles, data = generate_sinogram(data_size, angles)

    mask = np.zeros_like(origin_sinogram, dtype=np.bool)
    mask[:, mask.shape[1] // 2 - 40 : mask.shape[1] // 2 + 40] = True
    do_test(origin_sinogram, data, angles, mask, nitres=1000)

# def test_case_6():
#     angles = np.arange(0, 180, 0.1)
#     data_size = 256
#     ang_step = 30
#
#     origin_sinogram, angles, data = generate_sinogram(data_size, angles)
#     ideal_recon = astra_recon_2d_parallel(origin_sinogram[::ang_step], angles[::ang_step])
#     plt.figure(figsize=(7,7))
#     plt.imshow(ideal_recon, vmin=0, vmax=1, cmap=plt.cm.gray)
#     plt.show()
#
#     mask = np.zeros_like(origin_sinogram, dtype=np.bool)
#     num_angles = len(angles)
#
#     for i in range(ang_step):
#         mask[i*num_angles//ang_step: i*num_angles//ang_step+1, :] = True
#
#     do_test(origin_sinogram, data, angles, mask, nitres=1000, monitoring_iteration=100)


test_case_1()
test_case_2()
test_case_3()
test_case_4()
test_case_5()
