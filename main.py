import argparse
import re
import struct
import sys

import cv2
import numpy as np
import matplotlib.animation as ani
import matplotlib.pyplot as plt

def read_pfm(file):
    # Adopted from https://stackoverflow.com/questions/48809433/read-pfm-format-
    # in-python
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

def preprocess(imgLeft, dispLeft, imgRight, dispRight):
    img0, img1 = cv2.imread(imgLeft), cv2.imread(imgRight)
    if img0.size < img1.size:
        img1 = cv2.resize(img1, (img0.shape[1], img0.shape[0]))
    elif img1.size < img0.size:
        img0 = cv2.resize(img0, (img1.shape[1], img1.shape[0]))

    h, w, _ = img0.shape
    disp0 = cv2.resize(read_pfm(dispLeft), (w, h))
    disp1 = cv2.resize(read_pfm(dispRight), (w, h))

    # Remove INF and NAN from disparity maps
    for i in range(h):
        row0 = disp0[i]
        row1 = disp1[i]
        mask0 = np.isinf(row0) | np.isnan(row0)
        mask1 = np.isinf(row1) | np.isnan(row1)
        row0[mask0] = np.interp(mask0.nonzero()[0], 
                                (~mask0).nonzero()[0], row0[~mask0])
        row1[mask1] = np.interp(mask1.nonzero()[0], 
                                (~mask1).nonzero()[0], row1[~mask1])

    # Round to nearest pixel
    disp0 = np.round(disp0).astype(int)
    disp1 = np.round(disp1).astype(int)

    return img0, disp0, img1, disp1

def synthesize(alpha, imgLeft, dispLeft, imgRight, dispRight):
    # Load Images (imgLeft and imgRight assumed to have same shape)
    img0, img1 = imgLeft, imgRight
    h, w, c = img0.shape

    # Load disparity maps
    disp0, disp1 = dispLeft, dispRight
    alpha_disp0 = (disp0 * alpha).astype(int)
    alpha_disp1 = (disp1 * (1 - alpha)).astype(int)

    # Initialize arrays to hold warped disparities and images
    dL, dR = np.zeros_like(disp0), np.zeros_like(disp1)
    resL, resR = np.zeros_like(img0), np.zeros_like(img1)
    orig_indices = np.arange(w)
    for i in range(h):
        # Forward warp Left disparity map and Left image
        indices = orig_indices - alpha_disp0[i]
        valid = np.argwhere((indices < w) & (indices >= 0)).flatten()
        dL[i, indices[valid]] = disp0[i, valid]
        resL[i, indices[valid]] = img0[i, valid]

        # Forward warp Right disparity map and Right image
        indices = orig_indices + alpha_disp1[i]
        valid = np.argwhere((indices < w) & (indices >= 0)).flatten()
        dR[i, indices[valid]] = disp1[i, valid]
        resR[i, indices[valid]] = img1[i, valid]

    # Fill holes using other warped image
    res = np.zeros_like(img0)
    mask = np.repeat((dL >= dR)[..., np.newaxis], 3, axis=2)
    res[mask] = resL[mask]
    res[~mask] = resR[~mask]

    # Downsample image
    resized = res.copy()
    for _ in range(w // 1000):
        height, width, _ = resized.shape
        resized = np.delete(resized, list(range(0, height, 2)), axis=0)
        resized = np.delete(resized, list(range(0, width, 2)), axis=1)

    # Fill holes using Navier-Stokes inpainting
    mask = np.uint8(~np.any(resized, axis=2))
    resized = cv2.inpaint(resized, mask, 3, cv2.INPAINT_NS)

    # Upsample interpolated image and fill holes in final image
    upscaled = (cv2.resize(resized, (w, h))).astype(int)
    res[res == 0] = upscaled[res == 0]

    # Apply median blur to remove minor artifacts
    res = cv2.medianBlur(res, 3)

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_directory', type=str, 
                        help='path to image directory (e.g. images/adirondack)')
    parser.add_argument('alpha', type=float, nargs='?', default=0.5, 
                        help='default: 0.5 (halfway between the images)')
    parser.add_argument('-s', '--save', metavar='filename', nargs='?', type=str,
                        const='result.png',
                        help='flag to save results. default file is result.png')
    parser.add_argument('-p', '--pan', metavar='bound', type=float, nargs=2, 
                        help='pan camera from left bound to right bound')
    args = parser.parse_args()
    path = args.image_directory

    imgL = f'{path}/im0.png'
    imgR = f'{path}/im1.png'
    dispL = f'{path}/disp0.pfm'
    dispR = f'{path}/disp1.pfm'
    alpha = args.alpha

    imgL, dispL, imgR, dispR = preprocess(imgL, dispL, imgR, dispR)

    if not args.pan:
        res = synthesize(alpha, imgL, dispL, imgR, dispR)
        if args.save:
            cv2.imwrite(args.save, res)
        cv2.namedWindow(f'alpha = {alpha}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'alpha = {alpha}', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit()
    
    left = round(args.pan[0] * 10)
    right = round(args.pan[1] * 10)
    step_size = round((right - left) / 10)

    results = []
    for a in range(left, right + 1, step_size):
        results.append(synthesize(a / 10, imgL, dispL, imgR, dispR)[..., ::-1])

    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(len(results)):
        frames.append([plt.imshow(results[i], animated=True)])

    for i in range(len(frames) - 1, -1, -1):
        frames.append(frames[i])

    anim = ani.ArtistAnimation(fig, frames, interval=1, blit=True)

    if args.save:
        anim.save(args.save, writer=ani.PillowWriter(fps=10), dpi=150)

    plt.show()
