import math
import os
import pickle
import random

import cv2
import numpy as np
import torch

def is_npy_file(filename):
    return filename.endswith('.npy')

def _get_paths_from_dir(path):
    '''get numpy file path list from directory'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_npy_file(fname):
                file_path = os.path.join(dirpath, fname)
                files.append(file_path)
    assert files, '{:s} has no valid numpy file'.format(path)
    return files

def get_numpy_paths(data_type, dataroot):
    '''get numpy path list
    support numpy files'''
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'numpy':
            paths = sorted(_get_paths_from_dir(dataroot))
            return paths
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))

def read_numpy_file(path, min_value, max_value):
    '''read numpy file
    return: Numpy float32, 2D, grayscale, [0,1]'''
    data = np.load(path)
    assert data.ndim == 2, 'Data is not 2D'
    assert data.shape == (128, 128), 'Data is not 128x128 size'
    data = data.astype(np.float32) #/ 255.
    '''
    if min_value or max_value:
        if min_value == None:
            min_value = 0
        if max_value == None:
            max_value = 1
        data_bef = data
        data = (data - min_value) / (max_value - min_value)
        print(f'{data_bef.min():.3f} {data_bef.max():.3f} --> {data.min():.3f} {data.max():.3f}')
    '''   
    return data

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

# Functions
# matlab 'imresize' function, now only support 'bicubic'

def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3

    weight = (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (absx <= 2)).type_as(absx))
    return weight


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)

def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round
    is_numpy = False
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img.transpose(2, 0, 1)).cuda()
        is_numpy = True
    device = img.device
    # device = torch.device("cuda")

    is_batch = True
    if len(img.shape) == 3: # C, H, W
        img = img[None]
        is_batch = False

    B, in_C, in_H, in_W = img.size()
    img = img.view(-1, in_H, in_W)
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    # print(weights_H.device, indices_H.device, device)
    weights_H, indices_H = weights_H.to(device), indices_H.to(device)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W = weights_W.to(device), indices_W.to(device)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(B*in_C, in_H + sym_len_Hs + sym_len_He, in_W).to(device)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(B*in_C, out_H, in_W).to(device)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[:, i, :] = (img_aug[:, idx:idx + kernel_width, :].transpose(1, 2).matmul(
            weights_H[i][None,:,None].repeat(B*in_C,1, 1))).squeeze()

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(B*in_C, out_H, in_W + sym_len_Ws + sym_len_We).to(device)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long().to(device)
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(B*in_C, out_H, out_W).to(device)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, :, i] = (out_1_aug[:, :, idx:idx + kernel_width].matmul(
            weights_W[i][None,:,None].repeat(B*in_C, 1, 1))).squeeze()

    out_2 = out_2.contiguous().view(B, in_C, out_H, out_W)
    if not is_batch:
        out_2 = out_2[0]
    return out_2.cpu().numpy().transpose(1, 2, 0) if is_numpy else out_2

    ### Load data kernel map ###
def load_ker_map_list(path):
    real_ker_map_list = []
    batch_kermap = torch.load(path)
    size_kermap = batch_kermap.size()
    m = size_kermap[0]
    for i in range(m):
        real_ker_map_list.append(batch_kermap[i])

    return real_ker_map_list


if __name__ == '__main__':
    # Example usage
    dataroot = 'path_to_your_numpy_files'
    paths = get_numpy_paths(dataroot)
    for idx, path in enumerate(paths):
        data = read_numpy_file(path)
        print(f'Data shape: {data.shape}, Data range: {data.min()}-{data.max()}')

        # 결과 데이터를 npy 파일로 저장
        save_path = f'result_{idx}.npy'  # 저장할 파일 경로 및 이름 설정
        np.save(save_path, data)
        print(f'Saved processed data to {save_path}')
