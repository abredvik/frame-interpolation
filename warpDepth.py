import sys
import re
import struct
from skimage.io import imread, imshow
from skimage.transform import resize
import skimage
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, ndimage
import matplotlib.animation as ani
import time

def read_pfm(file):
    # Adopted from https://stackoverflow.com/questions/48809433/read-pfm-format-in-python
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = struct.unpack(fmt, buffer)
    return np.flipud(np.reshape(img, (height, width)))

def synthesize(alpha, imgLeft, dispLeft, imgRight, dispRight):
    # img0 = imread(imgLeft)
    img0 = imgLeft
    h, w, c = img0.shape
    # disp0 = resize(read_pfm(dispLeft), (h, w), preserve_range=True)
    disp0 = resize(dispLeft, (h, w), preserve_range=True)

    for i in range(h):
        row = disp0[i]
        mask = np.isinf(row) | np.isnan(row)
        row[mask] = np.interp(mask.nonzero()[0], (~mask).nonzero()[0], row[~mask])

    disp0 = np.round(disp0).astype(int)
    alpha_disp0 = (disp0 * alpha).astype(int)

    dL = np.zeros_like(disp0)
    resL = np.zeros_like(img0)
    orig_indices = np.arange(w)
    for i in range(h):
        indices = orig_indices - alpha_disp0[i]
        # valid = np.argwhere(indices >= 0).flatten()
        valid = np.argwhere((indices < w) & (indices >= 0)).flatten()
        dL[i, indices[valid]] = disp0[i, valid]
        resL[i, indices[valid]] = img0[i, valid]

    # flipDis = np.fliplr(dL)
    # mask = flipDis == 0
    # idx = np.where(~mask,np.arange(mask.shape[1]),0)
    # np.maximum.accumulate(idx,axis=1, out=idx)
    # flipDis[mask] = flipDis[np.nonzero(mask)[0], idx[mask]]

    # for k in range(3):
    #     flipRes = np.fliplr(resL[..., k])
    #     idx = np.where(~mask,np.arange(mask.shape[1]),0)
    #     np.maximum.accumulate(idx,axis=1, out=idx)
    #     flipRes[mask] = flipRes[np.nonzero(mask)[0], idx[mask]]

    # mask = dL == 0
    # idx = np.where(~mask,np.arange(mask.shape[1]),0)
    # np.maximum.accumulate(idx,axis=1, out=idx)
    # dL[mask] = dL[np.nonzero(mask)[0], idx[mask]]

    # for k in range(3):
    #     z = resL[..., k]
    #     idx = np.where(~mask,np.arange(mask.shape[1]),0)
    #     np.maximum.accumulate(idx,axis=1, out=idx)
    #     z[mask] = z[np.nonzero(mask)[0], idx[mask]]

    # img1 = imread(imgRight)
    img1 = imgRight
    h, w, c = img1.shape
    # disp1 = resize(read_pfm(dispRight), (h, w), preserve_range=True)
    disp1 = resize(dispRight, (h, w), preserve_range=True)

    for i in range(h):
        row = disp1[i]
        mask = np.isinf(row) | np.isnan(row)
        row[mask] = np.interp(mask.nonzero()[0], (~mask).nonzero()[0], row[~mask])

    disp1 = np.round(disp1).astype(int)
    alpha_disp1 = (disp1 * (1 - alpha)).astype(int)

    dR = np.zeros_like(disp1)
    resR = np.zeros_like(img1)
    orig_indices = np.arange(w)
    for i in range(h):
        indices = orig_indices + alpha_disp1[i]
        # valid = np.argwhere(indices < w).flatten()
        valid = np.argwhere((indices < w) & (indices >= 0)).flatten()
        dR[i, indices[valid]] = disp1[i, valid]
        resR[i, indices[valid]] = img1[i, valid]

    # mask = dR == 0
    # idx = np.where(~mask,np.arange(mask.shape[1]),0)
    # np.maximum.accumulate(idx,axis=1, out=idx)
    # dR[mask] = dR[np.nonzero(mask)[0], idx[mask]]

    # for k in range(3):
    #     z = resR[..., k]
    #     idx = np.where(~mask,np.arange(mask.shape[1]),0)
    #     np.maximum.accumulate(idx,axis=1, out=idx)
    #     z[mask] = z[np.nonzero(mask)[0], idx[mask]]

    # flipDis = np.fliplr(dR)
    # mask = flipDis == 0
    # idx = np.where(~mask,np.arange(mask.shape[1]),0)
    # np.maximum.accumulate(idx,axis=1, out=idx)
    # flipDis[mask] = flipDis[np.nonzero(mask)[0], idx[mask]]

    # for k in range(3):
    #     flipRes = np.fliplr(resR[..., k])
    #     idx = np.where(~mask,np.arange(mask.shape[1]),0)
    #     np.maximum.accumulate(idx,axis=1, out=idx)
    #     flipRes[mask] = flipRes[np.nonzero(mask)[0], idx[mask]]

    # resDisp = np.zeros_like(disp0)
    # resDisp[dL >= dR] = dL[dL >= dR]
    # resDisp[dL < dR] = dR[dL < dR]

    # fig, axes = plt.subplots(2,2)
    # axes[0,0].imshow(dL)
    # axes[0,1].imshow(dR)
    # axes[1,0].imshow(resL)
    # axes[1,1].imshow(resR)

    # # plt.imshow(np.concatenate([resL, resR], axis=1))
    # plt.show()

    res = np.zeros_like(img0)
    mask = np.repeat((dL >= dR)[..., np.newaxis], 3, axis=2)
    res[mask] = resL[mask]
    res[~mask] = resR[~mask]

    # resDepth = np.zeros_like(disp0)
    # mask = dL >= dR
    # resDepth[mask] = dL[mask]
    # resDepth[~mask] = dR[~mask]

    # plt.imshow(resDepth)
    # plt.show()

    # coords = np.argwhere(resDepth == 0)
    # for y, x in coords:
    #     if resDepth[y, x]:
    #         continue
    #     j = x
    #     while j + 3 < w and not np.all(resDepth[y, j:j+4]):
    #         j += 1
    #     if j == w or resDepth[y, x - 30] < resDepth[y, j]:
    #         resDepth[y, x:j] = resDepth[y, x - 30]
    #         res[y, x:j] = res[y, x - 30]
    #     else:
    #         resDepth[y, x:j] = resDepth[y, j]
    #         res[y, x:j] = res[y, j]
    
    # plt.imshow(resDepth)
    # plt.show()

    # plt.imshow(res)
    # plt.show()

    # disp03d = np.repeat(disp0[..., np.newaxis], 3, axis=2)
    # disp13d = np.repeat(disp1[..., np.newaxis], 3, axis=2)

    # mask1 = (res == 0) & (disp03d < disp13d)
    # mask2 = (res == 0) & (disp03d >= disp13d)
    # # print(res[zmask][dmask].shape)
    # res[mask1] = img0[mask1]
    # res[mask2] = img1[mask2]

    # res[res == 0] = (img0 * alpha + img1 * (1 - alpha)).astype(int)[res == 0]

    resized = res.copy()
    for _ in range(2):
        height, width, _ = resized.shape
        resized = np.delete(resized, list(range(0, height, 2)), axis=0)
        resized = np.delete(resized, list(range(0, width, 2)), axis=1)

    for k in range(3):
        z = resized[..., k]
        mask = z > 0
        y, x = np.where(mask)
        yi, xi = np.where(~mask)
        z[~mask] = interpolate.griddata((x, y), z[mask].ravel(), (xi, yi), method='nearest')

    upscaled = (resize(resized, (h, w), preserve_range=True)).astype(int)
    res[res == 0] = upscaled[res == 0]

    return res

adirondack = [imread('adirondack/im0.png'), read_pfm('adirondack/disp0.pfm'), 
              imread('adirondack/im1.png'), read_pfm('adirondack/disp1.pfm')]

jadeplant = [imread('jadeplant/im0.png'), read_pfm('jadeplant/disp0.pfm'), 
              imread('jadeplant/im1.png'), read_pfm('jadeplant/disp1.pfm')]

imgLeft, dispLeft, imgRight, dispRight = adirondack

# s = time.time()
# res = synthesize(0.5, imgLeft, dispLeft, imgRight, dispRight)
# e = time.time()
# print(e - s)
# plt.imshow(res)
# plt.show()

results = []
for a in range(-10, 21, 2):
    results.append(synthesize(a / 10, imgLeft, dispLeft, imgRight, dispRight))

frames = [] # for storing the generated images
fig = plt.figure()
for i in range(len(results)):
    frames.append([plt.imshow(results[i], animated=True)])

for i in range(len(frames) - 1, -1, -1):
    frames.append(frames[i])

anim = ani.ArtistAnimation(fig, frames, interval=1, blit=True)
# anim.save('fast.gif', writer=ani.PillowWriter(fps=10, bitrate=30000), dpi=200)
plt.show()
