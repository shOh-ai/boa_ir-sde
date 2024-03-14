import numpy as np
import torch
import os

def get_numpy_paths(dataroot):
    '''get numpy file path list from folder'''
    assert os.path.isdir(dataroot), f'{dataroot} is not a valid directory'
    numpy_files = []
    for dirpath, _, fnames in os.walk(dataroot):
        for fname in fnames:
            if fname.endswith('.npy'):
                numpy_path = os.path.join(dirpath, fname)
                numpy_files.append(numpy_path)
    assert numpy_files, f'{dataroot} has no valid numpy file'
    return numpy_files

def read_numpy_file(numpy_path):
    '''read numpy file'''
    return np.load(numpy_path)

def modcrop(img, scale):
    """
    Crop a 3D volume so that its dimensions are divisible by a scale factor
    img: 3D numpy array or 2D numpy array
    scale: scale factor
    """
    if img.ndim == 3:  # For 3D volumes
        D, H, W = img.shape
        D_r, H_r, W_r = D % scale, H % scale, W % scale
        img = img[:D - D_r, :H - H_r, :W - W_r]
    elif img.ndim == 2:  # For 2D images
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    else:
        raise ValueError("Unsupported image dimension.")
    return img

def numpy_to_tensor(img):
    """
    Convert a numpy array to a torch tensor.
    For 3D volumes, adds channel dimension at the beginning.
    """
    if img.ndim == 3:  # For 3D volumes
        img = np.expand_dims(img, axis=0)  # Add channel dimension (D, H, W) -> (1, D, H, W)
    elif img.ndim == 2:  # For 2D images
        img = np.expand_dims(img, axis=0)  # Add channel dimension (H, W) -> (1, H, W)
    img = np.expand_dims(img, axis=0)  # Add batch dimension for consistency (C, D, H, W) -> (1, C, D, H, W)
    return torch.from_numpy(img).float()

def tensor_to_numpy(tensor):
    """
    Convert a torch tensor to a numpy array.
    Assumes tensor is in CxHxW format for images or DxHxW for volumes.
    """
    return tensor.cpu().numpy().transpose(1, 2, 0) if tensor.dim() == 3 else tensor.cpu().numpy()

# Example function for resizing, assuming a function exists for 3D resizing
def resize_volume(img, scale_factor):
    """
    Resize a 3D volume using a specified scale factor.
    img: 3D numpy array
    scale_factor: tuple of 3 scale factors (D_scale, H_scale, W_scale)
    """
    # Placeholder for actual resize implementation
    # For example, using scipy.ndimage.zoom or a similar function
    return img  # Return resized image

# Example usage
if __name__ == "__main__":
    img = np.random.rand(100, 100, 100)  # Example 3D volume
    scale_factor = (0.5, 0.5, 0.5)  # Example scale factors
    resized_img = resize_volume(img, scale_factor)
    img_tensor = numpy_to_tensor(resized_img)
    print("Resized image shape:", img_tensor.shape)
