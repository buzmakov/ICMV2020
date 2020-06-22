# %%*- coding: utf-8 -*-
#% matplotlib inline

# %%
from tomopy.misc.phantom import shepp2d, shepp3d
import matplotlib
import numpy as np
import h5py
import pylab as plt
from tqdm import tqdm
from tomo.recon.astra_utils import astra_recon_2d_parallel, astra_fp_2d_parallel
from cv2 import medianBlur

matplotlib.rcParams.update({'font.size': 16})
# %%
data = np.squeeze(shepp2d(256)).astype('float32')
data /= data.max()
data = np.pad(data, data.shape[0]//4, mode='constant')
angles = np.arange(0, 180, 1)

sinogram = astra_fp_2d_parallel(data, angles)
rec = astra_recon_2d_parallel(sinogram, angles)

# %%
plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.imshow(data, cmap=plt.cm.gray)
plt.subplot(222)
plt.imshow(sinogram, cmap=plt.cm.gray)
plt.axis('tight')
plt.subplot(223)
plt.imshow(rec, cmap=plt.cm.gray)
plt.subplot(224)
plt.imshow(data-rec, cmap=plt.cm.seismic)
plt.show()


# %%
def get_x_cut(data):
    return np.arange(data.shape[0]) - data.shape[0] // 2, data[:, data.shape[1] // 2]


def get_y_cut(data):
    return np.arange(data.shape[1]) - data.shape[1] // 2, data[data.shape[0] // 2]

def create_circle_mask(size):
    X, Y = np.meshgrid(np.arange(size), np.arange(size))
    X -= size // 2
    Y -= size // 2
    mask = (X ** 2 + Y ** 2) < (size // 2) ** 2 - 10
    return mask

def create_masked_sinogram(sino, mask, mode):
    res = sino.copy()
    if mode == 'constant':
        res[mask==False] = 0
    return res

def fix_radon(sino, mask=None):
    if mask is None:
        radon_inv = sino.sum(axis=-1)
    else:
        radon_inv = (sino*mask).sum(axis=-1)/np.mean(mask.astype(np.float32), axis=-1)
    fixed_sino = sino.T/radon_inv*np.mean(radon_inv)
    return fixed_sino.T

def recon_with_mask(sinogram: np.ndarray, angles:np.ndarray, mask:np.ndarray, niters=5, method=[['FBP_CUDA']]):
    assert sinogram.shape == mask.shape
    circ_mask = create_circle_mask(sinogram.shape[1])
    t_sino = sinogram.copy()*mask
    rec = np.zeros_like(circ_mask)
    for _ in tqdm(range(niters)):
        # t_sino = fix_radon(t_sino, mask)
        rec = astra_recon_2d_parallel(t_sino, angles, method=method, data=rec)
        # rec *= circ_mask
        rec[rec<0] /= 2
        t_sino = astra_fp_2d_parallel(rec, angles)
        t_sino[mask] = sinogram[mask]
    return rec, t_sino,

mask = np.ones_like(sinogram, dtype=np.bool)
begin_stripe = mask.shape[1]//4+13
mask[:, begin_stripe:begin_stripe+5] = False
sino = create_masked_sinogram(sinogram, mask, 'constant')

mask_recon, res_sino = recon_with_mask(sino, angles, mask,
                                       niters=1000, method = [['SIRT_CUDA', 5]])

plt.figure(figsize=(15, 10))
plt.subplot(231)
plt.imshow(data, cmap=plt.cm.gray)
plt.subplot(232)
plt.imshow(sino, cmap=plt.cm.gray)
plt.colorbar(orientation='horizontal')
plt.subplot(233)
plt.imshow(res_sino, cmap=plt.cm.gray)
plt.colorbar(orientation='horizontal')

plt.subplot(234)
plt.imshow(mask_recon, cmap=plt.cm.gray)
plt.subplot(235)
plt.imshow(data-mask_recon, cmap=plt.cm.viridis)
plt.subplot(236)
plt.imshow(mask, cmap=plt.cm.gray)
plt.show()


# %%

for s_pad in tqdm(np.arange(1, 62, 4)):
    sinogram_cut = sinogram[:, s_pad:-s_pad]
    plt.figure(figsize=(10, 10))
    plt.imshow(sinogram_cut, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('tight')
    plt.title('Cut= {}'.format(s_pad))
    plt.show()

    rec_cut, proj_geom, cfg = astra_recon_2d_parallel(sinogram_cut, angles)

    plt.figure(figsize=(10, 10))
    plt.imshow(rec_cut, interpolation='nearest', cmap=plt.cm.gray)
    plt.title('Cut= {}'.format(s_pad))
    plt.show()

    # TODO: fixit
    padsize = s_pad * 2 * 2

    sinogram_padded = np.zeros((sinogram_cut.shape[0], sinogram_cut.shape[1] + padsize * 2), dtype='float32')
    sinogram_padded[:, padsize:-padsize] = sinogram_cut
    rec_pad0, proj_geom, cfg = astra_recon_2d_parallel(sinogram_padded, angles)

    plt.figure(figsize=(10, 10))
    plt.imshow(sinogram_padded, cmap=plt.cm.gray, interpolation='nearest')
    plt.title('Cut= {}'.format(s_pad))
    plt.axis('tight')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(rec_pad0, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.title('Cut= {}'.format(s_pad))
    plt.show()

    rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sinogram_padded, angles)
    sino = astra_fp_2d_parallel(rec_pad, angles)
    sino[:, padsize:-padsize] = sinogram_cut

    max_radon = sino.sum(axis=1).max()

    MU = rec.sum() * 2
    X, Y = np.meshgrid(np.arange(rec_pad.shape[0]), np.arange(rec_pad.shape[1]))

    X -= rec_pad.shape[0] // 2
    Y -= rec_pad.shape[1] // 2

    mask = (X ** 2 + Y ** 2) < (rec.shape[0] // 2) ** 2 - 10

    for i in tqdm(range(10000)):
        rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)
        rec_pad *= rec_pad > 0
        rec_pad *= mask
        sino = astra_fp_2d_parallel(rec_pad, angles)
        k = sino[:, padsize:-padsize].mean(axis=-1) / sinogram_cut.mean(axis=-1)
        if np.sum(np.argwhere(k == 0)) > 0:
            break

        sino[:, padsize:-padsize] = sinogram_cut
        sino[:, 0:padsize] = (sino[:, 0:padsize].T / k).T
        sino[:, -padsize:] = (sino[:, -padsize:].T / k).T

        sino = (sino.T / sino.sum(axis=1) * sino.sum(axis=1).mean()).T
        sino[:, padsize:-padsize] = sinogram_cut

    rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)

    plt.figure(figsize=(10, 10))
    plt.imshow(sino, cmap=plt.cm.gray, interpolation='nearest')
    plt.title('Cut= {}'.format(s_pad))
    plt.axis('tight')
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(rec_pad, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.title('Cut= {}'.format(s_pad))
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(*get_x_cut(rec_pad0), label='FBP')
    # plt.plot(*get_x_cut(rec_pad), label='rec_pad')
    plt.plot(*get_x_cut(rec_pad), label='Итеративный метод восстановления')
    plt.plot(*get_x_cut(rec), label='Исходный объект')
    plt.vlines([s_pad - 64, 64 - s_pad - 1], -0.1, 1.6, 'k', '--')
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
    plt.xlim(-100, 100)
    plt.ylabel('Поглощение')
    plt.xlabel('Номер канала детектора')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(data, cmap=plt.cm.gray)
    plt.title('а')
    plt.vlines([s_pad, 128 - s_pad], s_pad, 128 - s_pad, 'r', lw=2)
    plt.hlines([s_pad, 128 - s_pad], s_pad, 128 - s_pad, 'r', lw=2)

    plt.subplot(122)

    plt.imshow(rec_pad[rec_pad.shape[0] // 2 - rec.shape[0] // 2:rec_pad.shape[0] // 2 + rec.shape[0] // 2,
               rec_pad.shape[0] // 2 - rec.shape[0] // 2:rec_pad.shape[0] // 2 + rec.shape[0] // 2],
               interpolation='nearest', cmap=plt.cm.gray, vmax=1)

    plt.title('б')
    #     plt.colorbar(orientation='vertical')
    plt.vlines([s_pad, 128 - s_pad], s_pad, 128 - s_pad, 'r', lw=2)
    plt.hlines([s_pad, 128 - s_pad], s_pad, 128 - s_pad, 'r', lw=2)
    plt.show()

    plt.figure(figsize=(5, 5))

    plt.imshow(rec_pad0[rec_pad0.shape[0] // 2 - rec.shape[0] // 2:rec_pad0.shape[0] // 2 + rec.shape[0] // 2,
               rec_pad0.shape[0] // 2 - rec.shape[0] // 2:rec_pad0.shape[0] // 2 + rec.shape[0] // 2],
               interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)

    plt.title('в')

    plt.vlines([s_pad, 128 - s_pad], s_pad, 128 - s_pad, 'r', lw=2)
    plt.hlines([s_pad, 128 - s_pad], s_pad, 128 - s_pad, 'r', lw=2)
    plt.show()

# %%

for s_pad in tqdm(np.arange(32, 33, 1)):
    sinogram_cut = sinogram[:, s_pad:-s_pad]
    plt.figure(figsize=(10, 10))
    plt.imshow(sinogram_cut, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('tight')
    plt.title('Cut= {}'.format(s_pad))
    plt.show()

    rec_cut, proj_geom, cfg = astra_recon_2d_parallel(sinogram_cut, angles)

    plt.figure(figsize=(10, 10))
    plt.imshow(rec_cut, interpolation='nearest', cmap=plt.cm.gray)
    plt.title('Cut= {}'.format(s_pad))
    plt.show()

    # TODO: fixit
    padsize = s_pad * 2 * 2

    sinogram_padded = np.zeros((sinogram_cut.shape[0], sinogram_cut.shape[1] + padsize * 2), dtype='float32')
    sinogram_padded[:, padsize:-padsize] = sinogram_cut
    rec_pad0, proj_geom, cfg = astra_recon_2d_parallel(sinogram_padded, angles)

    plt.figure(figsize=(10, 10))
    plt.imshow(sinogram_padded, cmap=plt.cm.gray, interpolation='nearest')
    plt.title('Cut= {}'.format(s_pad))
    plt.axis('tight')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(rec_pad0, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.title('Cut= {}'.format(s_pad))
    plt.show()

    rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sinogram_padded, angles)
    sino = astra_fp_2d_parallel(rec_pad, angles)
    sino[:, padsize:-padsize] = sinogram_cut

    max_radon = sino.sum(axis=1).max()

    MU = rec.sum() * 2
    X, Y = np.meshgrid(np.arange(rec_pad.shape[0]), np.arange(rec_pad.shape[1]))

    X -= rec_pad.shape[0] // 2
    Y -= rec_pad.shape[1] // 2

    mask = (X ** 2 + Y ** 2) < (rec.shape[0] // 2) ** 2 - 10

    for i in tqdm(range(10001)):
        rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)
        rec_pad *= rec_pad > 0
        rec_pad *= mask
        sino = astra_fp_2d_parallel(rec_pad, angles)
        k = sino[:, padsize:-padsize].mean(axis=-1) / sinogram_cut.mean(axis=-1)
        if np.sum(np.argwhere(k == 0)) > 0:
            break

        sino[:, padsize:-padsize] = sinogram_cut
        sino[:, 0:padsize] = (sino[:, 0:padsize].T / k).T
        sino[:, -padsize:] = (sino[:, -padsize:].T / k).T

        sino = (sino.T / sino.sum(axis=1) * sino.sum(axis=1).mean()).T
        sino[:, padsize:-padsize] = sinogram_cut
        if i in [1, 5, 10, 100, 500, 1000, 5000, 10000]:
            rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)

            plt.figure(figsize=(20, 15))
            plt.subplot(221)
            plt.imshow(sino, cmap=plt.cm.gray, interpolation='nearest')
            plt.title('Iter= {}'.format(i))
            plt.axis('tight')
            #             plt.colorbar()
            #             plt.show()

            #             plt.subplot(221)
            #             plt.imshow(rec_pad, interpolation='nearest', cmap=plt.cm.gray,vmin=0, vmax=1)
            #             plt.title('Iter= {}'.format(i))
            #             plt.colorbar()
            #             plt.show()

            #             plt.figure(figsize=(10,5))
            plt.subplot(223)
            plt.plot(*get_x_cut(rec_pad0), 'g', label='FBP')
            # plt.plot(*get_x_cut(rec_pad), label='rec_pad')
            plt.plot(*get_x_cut(rec_pad), 'r', label='Итерационный алгоритм')
            plt.plot(*get_x_cut(rec), label='Исходный объект')
            plt.vlines([s_pad - 64, 64 - s_pad - 1], -0.1, 1.6, 'k', '--')
            plt.grid()
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
            plt.xlim(-100, 100)
            plt.ylabel('Поглощение')
            plt.xlabel('Номер канала детектора')
            #             plt.show()

            #             plt.figure(figsize=(10,5))
            #             plt.subplot(121)
            #             plt.imshow(data, cmap=plt.cm.gray)
            #             plt.title('а')
            #             plt.vlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)
            #             plt.hlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)

            plt.subplot(222)

            plt.imshow(rec_pad[rec_pad.shape[0] // 2 - rec.shape[0] // 2:rec_pad.shape[0] // 2 + rec.shape[0] // 2,
                       rec_pad.shape[0] // 2 - rec.shape[0] // 2:rec_pad.shape[0] // 2 + rec.shape[0] // 2],
                       interpolation='nearest', cmap=plt.cm.gray, vmax=1)

            plt.title('Итерационный алгоритм')
            #     plt.colorbar(orientation='vertical')
            #             plt.vlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)
            plt.hlines([64], 0, 128, 'r', lw=2)
            #             plt.show()

            #             plt.figure(figsize=(5,5))
            plt.subplot(224)
            plt.imshow(rec_pad0[rec_pad0.shape[0] // 2 - rec.shape[0] // 2:rec_pad0.shape[0] // 2 + rec.shape[0] // 2,
                       rec_pad0.shape[0] // 2 - rec.shape[0] // 2:rec_pad0.shape[0] // 2 + rec.shape[0] // 2],
                       interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)

            plt.title('FBP')

            #             plt.vlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)
            #             plt.hlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)
            plt.hlines([64], 0, 128, 'g', lw=2)
            plt.show()
# %%

for s_pad in tqdm(np.arange(32, 33, 1)):
    sinogram_cut = sinogram[:, s_pad:-s_pad]
    plt.figure(figsize=(10, 10))
    plt.imshow(sinogram_cut, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('tight')
    plt.title('Cut= {}'.format(s_pad))
    plt.show()

    rec_cut, proj_geom, cfg = astra_recon_2d_parallel(sinogram_cut, angles)

    plt.figure(figsize=(10, 10))
    plt.imshow(rec_cut, interpolation='nearest', cmap=plt.cm.gray)
    plt.title('Cut= {}'.format(s_pad))
    plt.show()

    # TODO: fixit
    padsize = s_pad * 2 * 2

    sinogram_padded = np.zeros((sinogram_cut.shape[0], sinogram_cut.shape[1] + padsize * 2), dtype='float32')
    sinogram_padded[:, padsize:-padsize] = sinogram_cut
    rec_pad0, proj_geom, cfg = astra_recon_2d_parallel(sinogram_padded, angles)

    plt.figure(figsize=(10, 10))
    plt.imshow(sinogram_padded, cmap=plt.cm.gray, interpolation='nearest')
    plt.title('Cut= {}'.format(s_pad))
    plt.axis('tight')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(rec_pad0, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.title('Cut= {}'.format(s_pad))
    plt.show()

    rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sinogram_padded, angles)
    sino = astra_fp_2d_parallel(rec_pad, angles)
    sino[:, padsize:-padsize] = sinogram_cut

    max_radon = sino.sum(axis=1).max()

    MU = rec.sum() * 2
    X, Y = np.meshgrid(np.arange(rec_pad.shape[0]), np.arange(rec_pad.shape[1]))

    X -= rec_pad.shape[0] // 2
    Y -= rec_pad.shape[1] // 2

    mask = (X ** 2 + Y ** 2) < (rec.shape[0] // 2) ** 2 - 10

    for i in tqdm(range(100)):
        rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)
        rec_pad *= rec_pad > 0
        #         rec_pad*=mask
        sino = astra_fp_2d_parallel(rec_pad, angles)
        k = sino[:, padsize:-padsize].mean(axis=-1) / sinogram_cut.mean(axis=-1)
        if np.sum(np.argwhere(k == 0)) > 0:
            break

        sino[:, padsize:-padsize] = sinogram_cut
        sino[:, 0:padsize] = (sino[:, 0:padsize].T / k).T
        sino[:, -padsize:] = (sino[:, -padsize:].T / k).T

        sino = (sino.T / sino.sum(axis=1) * sino.sum(axis=1).mean()).T
        sino[:, padsize:-padsize] = sinogram_cut
        if i in [1, 5, 10, 100, 500, 1000, 5000, 10000]:
            rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)

            plt.figure(figsize=(20, 15))
            plt.subplot(221)
            plt.imshow(sino, cmap=plt.cm.gray, interpolation='nearest')
            plt.title('Iter= {}'.format(i))
            plt.axis('tight')
            #             plt.colorbar()
            #             plt.show()

            #             plt.subplot(221)
            #             plt.imshow(rec_pad, interpolation='nearest', cmap=plt.cm.gray,vmin=0, vmax=1)
            #             plt.title('Iter= {}'.format(i))
            #             plt.colorbar()
            #             plt.show()

            #             plt.figure(figsize=(10,5))
            plt.subplot(223)
            plt.plot(*get_x_cut(rec_pad0), 'g', label='FBP')
            # plt.plot(*get_x_cut(rec_pad), label='rec_pad')
            plt.plot(*get_x_cut(rec_pad), 'r', label='Итерационный алгоритм')
            plt.plot(*get_x_cut(rec), label='Исходный объект')
            plt.vlines([s_pad - 64, 64 - s_pad - 1], -0.1, 1.6, 'k', '--')
            plt.grid()
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
            plt.xlim(-100, 100)
            plt.ylabel('Поглощение')
            plt.xlabel('Номер канала детектора')
            #             plt.show()

            #             plt.figure(figsize=(10,5))
            #             plt.subplot(121)
            #             plt.imshow(data, cmap=plt.cm.gray)
            #             plt.title('а')
            #             plt.vlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)
            #             plt.hlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)

            plt.subplot(222)

            plt.imshow(rec_pad[rec_pad.shape[0] // 2 - rec.shape[0] // 2:rec_pad.shape[0] // 2 + rec.shape[0] // 2,
                       rec_pad.shape[0] // 2 - rec.shape[0] // 2:rec_pad.shape[0] // 2 + rec.shape[0] // 2],
                       interpolation='nearest', cmap=plt.cm.gray, vmax=1)

            plt.title('Итерационный алгоритм')
            #     plt.colorbar(orientation='vertical')
            #             plt.vlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)
            plt.hlines([64], 0, 128, 'r', lw=2)
            #             plt.show()

            #             plt.figure(figsize=(5,5))
            plt.subplot(224)
            plt.imshow(rec_pad0[rec_pad0.shape[0] // 2 - rec.shape[0] // 2:rec_pad0.shape[0] // 2 + rec.shape[0] // 2,
                       rec_pad0.shape[0] // 2 - rec.shape[0] // 2:rec_pad0.shape[0] // 2 + rec.shape[0] // 2],
                       interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)

            plt.title('FBP')

            #             plt.vlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)
            #             plt.hlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)
            plt.hlines([64], 0, 128, 'g', lw=2)
            plt.show()

for s_pad in tqdm(np.arange(32, 33, 1)):
    sinogram_cut = sinogram[:, s_pad:-s_pad]
    plt.figure(figsize=(10, 10))
    plt.imshow(sinogram_cut, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('tight')
    plt.title('Cut= {}'.format(s_pad))
    plt.show()

    rec_cut, proj_geom, cfg = astra_recon_2d_parallel(sinogram_cut, angles)

    plt.figure(figsize=(10, 10))
    plt.imshow(rec_cut, interpolation='nearest', cmap=plt.cm.gray)
    plt.title('Cut= {}'.format(s_pad))
    plt.show()

    # TODO: fixit
    padsize = s_pad * 2 * 2

    sinogram_padded = np.zeros((sinogram_cut.shape[0], sinogram_cut.shape[1] + padsize * 2), dtype='float32')
    sinogram_padded[:, padsize:-padsize] = sinogram_cut
    rec_pad0, proj_geom, cfg = astra_recon_2d_parallel(sinogram_padded, angles)

    plt.figure(figsize=(10, 10))
    plt.imshow(sinogram_padded, cmap=plt.cm.gray, interpolation='nearest')
    plt.title('Cut= {}'.format(s_pad))
    plt.axis('tight')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(rec_pad0, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)
    plt.title('Cut= {}'.format(s_pad))
    plt.show()

    rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sinogram_padded, angles)
    sino = astra_fp_2d_parallel(rec_pad, angles)
    sino[:, padsize:-padsize] = sinogram_cut

    max_radon = sino.sum(axis=1).max()

    MU = rec.sum() * 2
    X, Y = np.meshgrid(np.arange(rec_pad.shape[0]), np.arange(rec_pad.shape[1]))

    X -= rec_pad.shape[0] // 2
    Y -= rec_pad.shape[1] // 2

    mask = (X ** 2 + Y ** 2) < (rec.shape[0] // 2) ** 2 - 10

    for i in tqdm(range(10001)):
        rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)
        rec_pad *= rec_pad > 0
        rec_pad *= mask
        sino = astra_fp_2d_parallel(rec_pad, angles)
        k = sino[:, padsize:-padsize].mean(axis=-1) / sinogram_cut.mean(axis=-1)
        if np.sum(np.argwhere(k == 0)) > 0:
            break

        sino[:, padsize:-padsize] = sinogram_cut
        #         sino[:,0:padsize]=(sino[:,0:padsize].T/k).T
        #         sino[:,-padsize:]=(sino[:,-padsize:].T/k).T

        #         sino = (sino.T/sino.sum(axis=1)*sino.sum(axis=1).mean()).T
        sino[:, padsize:-padsize] = sinogram_cut
        if i in [10000, ]:
            rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)

            plt.figure(figsize=(20, 15))
            plt.subplot(221)
            plt.imshow(sino, cmap=plt.cm.gray, interpolation='nearest')
            plt.title('Iter= {}'.format(i))
            plt.axis('tight')
            #             plt.colorbar()
            #             plt.show()

            #             plt.subplot(221)
            #             plt.imshow(rec_pad, interpolation='nearest', cmap=plt.cm.gray,vmin=0, vmax=1)
            #             plt.title('Iter= {}'.format(i))
            #             plt.colorbar()
            #             plt.show()

            #             plt.figure(figsize=(10,5))
            plt.subplot(223)
            plt.plot(*get_x_cut(rec_pad0), 'g', label='FBP')
            # plt.plot(*get_x_cut(rec_pad), label='rec_pad')
            plt.plot(*get_x_cut(rec_pad), 'r', label='Итерационный алгоритм')
            plt.plot(*get_x_cut(rec), label='Исходный объект')
            plt.vlines([s_pad - 64, 64 - s_pad - 1], -0.1, 1.6, 'k', '--')
            plt.grid()
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
            plt.xlim(-100, 100)
            plt.ylabel('Поглощение')
            plt.xlabel('Номер канала детектора')
            #             plt.show()

            #             plt.figure(figsize=(10,5))
            #             plt.subplot(121)
            #             plt.imshow(data, cmap=plt.cm.gray)
            #             plt.title('а')
            #             plt.vlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)
            #             plt.hlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)

            plt.subplot(222)

            plt.imshow(rec_pad[rec_pad.shape[0] // 2 - rec.shape[0] // 2:rec_pad.shape[0] // 2 + rec.shape[0] // 2,
                       rec_pad.shape[0] // 2 - rec.shape[0] // 2:rec_pad.shape[0] // 2 + rec.shape[0] // 2],
                       interpolation='nearest', cmap=plt.cm.gray, vmax=1)

            plt.title('Итерационный алгоритм')
            #     plt.colorbar(orientation='vertical')
            #             plt.vlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)
            plt.hlines([64], 0, 128, 'r', lw=2)
            #             plt.show()

            #             plt.figure(figsize=(5,5))
            plt.subplot(224)
            plt.imshow(rec_pad0[rec_pad0.shape[0] // 2 - rec.shape[0] // 2:rec_pad0.shape[0] // 2 + rec.shape[0] // 2,
                       rec_pad0.shape[0] // 2 - rec.shape[0] // 2:rec_pad0.shape[0] // 2 + rec.shape[0] // 2],
                       interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)

            plt.title('FBP')

            #             plt.vlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)
            #             plt.hlines([s_pad,128-s_pad],s_pad,128-s_pad, 'r', lw=2)
            plt.hlines([64], 0, 128, 'g', lw=2)
            plt.show()

# %%
# plt.figure(figsize=(10,10))
# plt.imshow(sinogram_cut, cmap=plt.cm.gray, interpolation='nearest')
# plt.colorbar(orientation='horizontal')
# plt.axis('tight')
# plt.show()

# rec_cut, proj_geom, cfg = astra_recon_2d_parallel(sinogram_cut, angles)

# plt.figure(figsize=(10,10))
# # plt.imshow(rec,vmin=0.1, vmax=0.2)
# plt.imshow(rec_cut, interpolation='nearest', cmap=plt.cm.gray)
# plt.colorbar(orientation='horizontal')
# plt.show()

# %%
# padsize = s_pad*2*2

# sinogram_padded = np.zeros((sinogram_cut.shape[0],sinogram_cut.shape[1]+padsize*2), dtype='float32')
# sinogram_padded[:,padsize:-padsize] = sinogram_cut
# rec_pad0, proj_geom, cfg = astra_recon_2d_parallel(sinogram_padded, angles)

# %%
# plt.figure(figsize=(10,10))
# plt.imshow(sinogram_padded, cmap=plt.cm.gray, interpolation='nearest')
# plt.colorbar(orientation='horizontal')
# plt.axis('tight')
# plt.show()

# plt.figure(figsize=(10,10))
# # plt.imshow(rec,vmin=0.1, vmax=0.2)
# plt.imshow(rec_pad0, interpolation='nearest', cmap=plt.cm.gray, vmax=1)
# plt.colorbar(orientation='horizontal')
# plt.show()

# %%
import scipy.ndimage


def my_rc(sino0, level):
    def get_my_b(level):
        t = np.mean(sino0, axis=0)
        gt = scipy.ndimage.filters.gaussian_filter1d(t, level / 2.)
        return gt - t

    def get_my_a(level):
        my_b = get_my_b(level)
        return np.mean(my_b) / my_b.shape[0]

    my_a = get_my_a(level)
    my_b = get_my_b(level)

    res = sino0.copy()
    if not level == 0:
        res += sino0 * my_a + my_b

    return res


# %%
# rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sinogram_padded, angles)
# sino = astra_fp_2d_parallel(rec_pad, angles)
# sino[:,padsize:-padsize] = sinogram_cut

# max_radon=sino.sum(axis=1).max()

# MU = rec.sum()*2
# X,Y = np.meshgrid(np.arange(rec_pad.shape[0]),np.arange(rec_pad.shape[1]))

# X-=rec_pad.shape[0]//2
# Y-=rec_pad.shape[1]//2

# mask = (X**2+Y**2)<(rec.shape[0]//2)**2-10

# # for i in tqdm(range(1000)):
# #     rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)

# #     rec_pad*=rec_pad>0
# #     rec_pad*=mask
# #     rec_pad[rec_pad>1] = 1
# # #     if rec_pad.sum()>MU:
# # #         rec_pad = rec_pad/rec_pad.sum()*MU
# #     sino = astra_fp_2d_parallel(rec_pad, angles)

# #     if i < 150:
# #         sino = my_rc(sino, 150-i)

# #     sino[:,padsize:-padsize] = sinogram_cut

# for i in tqdm(range(1000)):
#     rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)
#     rec_pad*=rec_pad>0
#     rec_pad*=mask
# #     rec_pad[rec_pad>1] = 1
#     sino = astra_fp_2d_parallel(rec_pad, angles)

# #     sino = (sino.T/sino.sum(axis=1)*sinogram_cut.mean()).T
# #   t = sino[:,pad_size:-shift-pad_size]
# #     sino[:,padsize:-padsize] = (sinogram_cut+sino[:,padsize:-padsize])/2.
#     sino[:,padsize:-padsize] = sinogram_cut
#     sino = (sino.T/sino.sum(axis=1)*max_radon*1.4).T
#     sino[:,padsize:-padsize] = sinogram_cut

# rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)

# plt.figure(figsize=(10,10))
# plt.imshow(sino, cmap=plt.cm.gray, interpolation='nearest')
# plt.colorbar(orientation='horizontal')
# plt.axis('tight')
# plt.show()

# plt.figure(figsize=(10,10))
# # plt.imshow(rec,vmin=0.1, vmax=0.2)
# plt.imshow(rec_pad, interpolation='nearest', cmap=plt.cm.gray,vmax=1)
# plt.colorbar(orientation='horizontal')
# plt.show()

# %%

# # rec_plot = np.zeros_like(rec_pad)
# # rec_plot[150:-150,150:-150] = data
# def get_x_cut(data):
#     return np.arange(data.shape[0])-data.shape[0]//2, data[:,data.shape[1]//2]

# def get_y_cut(data):
#     return np.arange(data.shape[1])-data.shape[1]//2, data[data.shape[0]//2]

# plt.figure(figsize=(10,5))
# plt.plot(*get_x_cut(rec_pad0), label='rec_pad0')
# plt.plot(*get_x_cut(rec_pad), label='rec_pad')
# plt.plot(*get_x_cut(rec_cut), label='rec_cut')
# plt.plot(*get_x_cut(rec), label='rec')
# plt.grid()
# plt.legend(loc=0)
# plt.xlim(-100,100)
# plt.show()

# plt.figure(figsize=(10,5))
# plt.plot(*get_x_cut(rec_pad0), label='FBP')
# # plt.plot(*get_x_cut(rec_pad), label='rec_pad')
# plt.plot(*get_x_cut(rec_pad), label='Итеративный метод восстановления')
# plt.plot(*get_x_cut(rec), label='Исхдный объект')
# plt.vlines([-32,32],-0.1,1.6, 'k', '--')
# plt.grid()
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
# plt.xlim(-100,100)
# plt.ylabel('Поглощение')
# plt.xlabel('Номер канала детектора')
# plt.show()

# plt.figure(figsize=(10,5))
# plt.plot(*get_y_cut(rec_pad0), label='rec_pad0')
# plt.plot(*get_y_cut(rec_pad), label='rec_pad')
# plt.plot(*get_y_cut(rec_cut), label='rec_cut')
# plt.plot(*get_y_cut(rec), label='rec')
# plt.grid()
# plt.legend(loc=0)
# plt.xlim(-100,100)
# plt.show()


# %%
rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)

plt.figure(figsize=(10, 10))
plt.imshow(sino[:, padsize - s_pad:-padsize + s_pad], cmap=plt.cm.gray, interpolation='nearest')
plt.colorbar(orientation='horizontal')
plt.axis('tight')
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(sino[0, padsize - s_pad:-padsize + s_pad])
plt.plot(sino[sino.shape[1] // 2, padsize - s_pad:-padsize + s_pad])
plt.plot(sino[-1, padsize - s_pad:-padsize + s_pad])
plt.plot(sinogram[0, :], '--')
plt.plot(sinogram[0, :], '--')
plt.plot(sinogram[sino.shape[1] // 2, :], '--')
plt.plot(sinogram[-1, :], '--')

plt.grid()
plt.show()

plt.figure(figsize=(10, 10))
# plt.imshow(rec,vmin=0.1, vmax=0.2)
plt.imshow(rec_pad[rec_pad.shape[0] // 2 - rec.shape[0] // 2:rec_pad.shape[0] // 2 + rec.shape[0] // 2,
           rec_pad.shape[0] // 2 - rec.shape[0] // 2:rec_pad.shape[0] // 2 + rec.shape[0] // 2],
           interpolation='nearest', cmap=plt.cm.gray, vmax=1)
plt.colorbar(orientation='horizontal')
plt.show()

plt.figure(figsize=(10, 10))
# plt.imshow(rec,vmin=0.1, vmax=0.2)
plt.imshow(rec_pad0[rec_pad0.shape[0] // 2 - rec.shape[0] // 2:rec_pad0.shape[0] // 2 + rec.shape[0] // 2,
           rec_pad0.shape[0] // 2 - rec.shape[0] // 2:rec_pad0.shape[0] // 2 + rec.shape[0] // 2],
           interpolation='nearest', cmap=plt.cm.gray, vmax=1)
plt.colorbar(orientation='horizontal')
plt.show()

plt.figure(figsize=(10, 10))
# plt.imshow(rec,vmin=0.1, vmax=0.2)
plt.imshow(rec_cut,
           interpolation='nearest', cmap=plt.cm.gray, vmax=1)
plt.colorbar(orientation='horizontal')
plt.show()

# %%
import matplotlib

matplotlib.rcParams.update({'font.size': 22})

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(data, cmap=plt.cm.gray)
# plt.vlines([32,96],32,96, 'r', lw=2)
# plt.hlines([32,96],32,96, 'r', lw=2)
plt.title('а')
# plt.imshow(rec_pad[rec_pad.shape[0]//2-rec.shape[0]//2:rec_pad.shape[0]//2+rec.shape[0]//2,
#                   rec_pad.shape[0]//2-rec.shape[0]//2:rec_pad.shape[0]//2+rec.shape[0]//2],
#            interpolation='nearest', cmap=plt.cm.gray,vmax=1)
plt.colorbar(orientation='vertical')
# plt.show()

plt.subplot(122)
# plt.imshow(rec,vmin=0.1, vmax=0.2)
plt.imshow(rec_pad[rec_pad.shape[0] // 2 - rec.shape[0] // 2:rec_pad.shape[0] // 2 + rec.shape[0] // 2,
           rec_pad.shape[0] // 2 - rec.shape[0] // 2:rec_pad.shape[0] // 2 + rec.shape[0] // 2],
           interpolation='nearest', cmap=plt.cm.gray, vmax=1)
# plt.vlines([32,96],32,96, 'r', lw=2)
# plt.hlines([32,96],32,96, 'r', lw=2)
plt.title('б')
plt.colorbar(orientation='vertical')
plt.show()

plt.figure(figsize=(5, 5))
# plt.subplot(121)
# plt.imshow(rec,vmin=0.1, vmax=0.2)
plt.imshow(rec_pad0[rec_pad0.shape[0] // 2 - rec.shape[0] // 2:rec_pad0.shape[0] // 2 + rec.shape[0] // 2,
           rec_pad0.shape[0] // 2 - rec.shape[0] // 2:rec_pad0.shape[0] // 2 + rec.shape[0] // 2],
           interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)
# plt.vlines([32,96],32,96, 'r', lw=2)
# plt.hlines([32,96],32,96, 'r', lw=2)
plt.title('в')
plt.colorbar(orientation='vertical')
plt.show()

# %%
x = np.zeros((128 * 3 // 2, 180))
for t in range(180):
    x[int(64 * 3 // 2 + 64 * np.sin((t + 30) / 180. * np.pi)), t:t + 2] = 1

plt.figure(figsize=(10, 10))
plt.imshow(x, cmap=plt.cm.gray)
# %%

data_file = '/home/makov/diskmnt/big/robotom/8381fee8-a5cb-41d5-b6e5-4f5617458b46/tomo_rec.h5'

# %%

with h5py.File(data_file) as h5f:
    data = h5f['Reconstruction'][1200]
# %%

data[data < 0] = 0

plt.figure(figsize=(10, 10))
plt.imshow(data, cmap=plt.cm.gray)
# plt.colorbar()

angles = np.arange(0, 180, 1. / 10) * np.pi / 180 + 0.3
sinogram = astra_fp_2d_parallel(data, angles)

plt.figure(figsize=(10, 10))
plt.imshow(sinogram, cmap=plt.cm.gray)
# plt.colorbar()

# %%
padsize = 300
s_pad = padsize

sinogram_cut = sinogram[:, padsize:-padsize]
plt.figure(figsize=(10, 10))
plt.imshow(sinogram_cut, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('tight')
plt.title('Cut= {}'.format(s_pad))
plt.show()

rec_cut, proj_geom, cfg = astra_recon_2d_parallel(sinogram_cut, angles)

plt.figure(figsize=(10, 10))
plt.imshow(rec_cut, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Cut= {}'.format(s_pad))
plt.show()

# TODO: fixit
# padsize = s_pad*2*2

sinogram_padded = np.zeros((sinogram_cut.shape[0], sinogram_cut.shape[1] + padsize * 2), dtype='float32')
sinogram_padded[:, padsize:-padsize] = sinogram_cut
rec_pad0, proj_geom, cfg = astra_recon_2d_parallel(sinogram_padded, angles)

plt.figure(figsize=(10, 10))
plt.imshow(sinogram_padded, cmap=plt.cm.gray, interpolation='nearest')
plt.title('Cut= {}'.format(s_pad))
plt.axis('tight')
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(rec_pad0, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)
plt.title('Cut= {}'.format(s_pad))
plt.show()

rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sinogram_padded, angles)
sino = astra_fp_2d_parallel(rec_pad, angles)
sino[:, padsize:-padsize] = sinogram_cut

# # max_radon=sino.sum(axis=1).max()

# MU = rec.sum()*2
X, Y = np.meshgrid(np.arange(rec_pad.shape[0]), np.arange(rec_pad.shape[1]))

X -= rec_pad.shape[0] // 2
Y -= rec_pad.shape[1] // 2

mask = (X ** 2 + Y ** 2) < (rec_pad.shape[0] // 2 - 10) ** 2  # Fix it in the up !

for i in tqdm(range(5000)):
    rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)
    rec_pad *= rec_pad > 0
    rec_pad *= mask
    sino = astra_fp_2d_parallel(rec_pad, angles)
    k = sino[:, padsize:-padsize].mean(axis=-1) / sinogram_cut.mean(axis=-1)
    if np.sum(np.argwhere(k == 0)) > 0:
        break

    sino[:, padsize:-padsize] = sinogram_cut
    sino[:, 0:padsize] = (sino[:, 0:padsize].T / k).T
    sino[:, -padsize:] = (sino[:, -padsize:].T / k).T

    sino = (sino.T / sino.sum(axis=1) * sino.sum(axis=1).mean()).T
    sino = medianBlur(sino, 5)  # # Fix it in the up !
    #     sino[sino>200]=200
    sino[:, padsize:-padsize] = sinogram_cut

rec_pad, proj_geom, cfg = astra_recon_2d_parallel(sino, angles)

plt.figure(figsize=(10, 10))
plt.imshow(sino, cmap=plt.cm.gray, interpolation='nearest')
plt.title('Cut= {}'.format(s_pad))
plt.axis('tight')
plt.colorbar()
plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(rec_pad, interpolation='nearest', cmap=plt.cm.gray, vmin=0, vmax=1)
plt.title('Cut= {}'.format(s_pad))
# plt.colorbar()
plt.show()
# %%


