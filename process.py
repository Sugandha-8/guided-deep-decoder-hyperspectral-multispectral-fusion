from scipy.io import loadmat
import numpy as np
from scipy.ndimage import zoom
import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from spectral import io
from scipy.ndimage import zoom
from scipy.io import savemat
from spectral import open_image
from tifffile import imread
from scipy.ndimage import gaussian_filter
from PIL import Image


matHS_data = loadmat('/scratch/r.sugandha/test_deep/Registered_data/Location1/frt000066a4_07_if166j_mtr3_registered.mat')

# Step : Access the data; adjust 'data_key' to match the key in your .mat file
data_key_HS = 'imgHS_cropped'  # Replace with the correct key
hs_image = np.array(matHS_data[data_key_HS])

# This assumes hs_image is a 3D array (e.g., spatial x spatial x spectral/bands)
downsampling_factor = (32 / hs_image.shape[0], 32 / hs_image.shape[1], 1)

# Step : Perform downsampling
downsampled_hs_image = zoom(hs_image, zoom=downsampling_factor, order=1)  # Using bilinear interpolation

# Verifying the operation
print("Original reg shape:", hs_image.shape)
print("Downsampled shape:", downsampled_hs_image.shape)

### MS ####
matMS_data = loadmat('/scratch/r.sugandha/test_deep/Registered_data/Location1/D14_032794_1989_XN_18N282W_registered.mat')


data_key_MS = 'imgMS_cat'  # Replace with the correct key
ms_image = np.array(matMS_data[data_key_MS])


print("Original shape:", ms_image.shape)

def gaussian_downsampling(data, options):
    h_rows, h_cols = options['hRows'], options['hCols']
    p_rows, p_cols = options['pRows'], options['pCols']
    hsize = (round(p_rows / h_rows), round(p_cols / h_cols))
    sigma = hsize[0] / (2 * np.sqrt(2 * np.log(2)))
    data = gaussian_filter(data, sigma=sigma)
    if data.dtype != np.uint8:
        data = (data - data.min()) / (data.max() - data.min()) * 255
        data = data.astype(np.uint8)

    img = Image.fromarray(data)
   

    # Resize the image using Pillow
    resized_img = img.resize((h_cols, h_rows), Image.LANCZOS)  # Using LANCZOS for high-quality downsampling

    # Convert back to numpy array if necessary
    resized_data = np.array(resized_img)
    return resized_data

def estimate_srf(hcube, pcube, options):
    lr_pan = gaussian_downsampling(pcube, options).reshape(-1, options['pBands'])
    hcube = hcube.reshape(-1, options['hBands'])
    hcube = np.hstack([hcube, np.ones((hcube.shape[0], 1))])
    srf = np.linalg.lstsq(hcube, lr_pan, rcond=None)[0].T
    return srf


hs_cube = downsampled_hs_image



#print("true",hs_cube == 65535)
is_65535_present = np.any(hs_cube == 65535)
print(f"65535 present in the data: {is_65535_present}")
hs_cube[hs_cube == 65535] = 0
print("hs cube shape",hs_cube.shape)
# Load multispectral image
ms_cube = ms_image
print("ms cube shape",ms_cube.shape)
#imread('F21_044125_2056_XN_25N021W.tiff')

# Define parameters
parameters = {
    'hRows': hs_cube.shape[0], 
    'hCols': hs_cube.shape[1], 
    'hBands': hs_cube.shape[2],
    'pRows': ms_cube.shape[0], 
    'pCols': ms_cube.shape[1], 
    'pBands': ms_cube.shape[2]
}

# Estimate Spectral Response Function
srf = estimate_srf(hs_cube, ms_cube, parameters)
print("srf shape ",srf.shape)


# # Dropping the last column of the SRF matrix
srf = srf[:, :-1]
print("srf shape new",srf.shape)


data_to_save = {
     'MSI': ms_image,  # Assuming you have this from your image processing
     'HSI': downsampled_hs_image,  # Assuming you have this from your image processing
     'R': srf
 }


savemat('/scratch/r.sugandha/test_deep/guided-deep-decoder/GDD_code/my_data_07_if166.mat', data_to_save)
