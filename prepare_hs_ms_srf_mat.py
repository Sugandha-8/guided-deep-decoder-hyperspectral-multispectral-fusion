import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from spectral import io
from scipy.ndimage import zoom, gaussian_filter
from scipy.io import savemat, loadmat
from spectral import open_image
from tifffile import imread
from PIL import Image



# Example CTX TIFF path
# /scratch/r.sugandha/test_deep/B17_016279_2420_XN_62N130W.tiff   # 168 one
# /scratch/r.sugandha/test_deep/guided-deep-decoder/GDD_code/F21_044125_2056_XN_25N021W.tiff  # 166 one

with rasterio.open('/scratch/r.sugandha/test_deep/guided-deep-decoder/GDD_code/F21_044125_2056_XN_25N021W.tiff') as dataset:
    # Basic info
    print("Shape: (height, width) =", dataset.shape)   # (H, W)
    print("Number of CTX bands:", dataset.count)       # number of bands

    # Example: read a single band just to inspect
    band2 = dataset.read(2)
    print("Shape of band 2:", band2.shape)

    # Downsample CTX to 1024 × 1024 using bilinear resampling
    scale_height = dataset.height / 1024
    scale_width = dataset.width / 1024

    data_downsampled = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height / scale_height),
            int(dataset.width / scale_width)
        ),
        resampling=Resampling.bilinear
    )

# Convert to (H, W, C)
ctx_image = np.array(data_downsampled)
ctx_image = ctx_image.transpose((1, 2, 0))
print("Downsampled CTX shape:", ctx_image.shape)
print("min ctx_image:", np.min(ctx_image))
print("max ctx_image:", np.max(ctx_image))

# Visualize one CTX band (optional)
plt.imshow(data_downsampled[1], cmap='gray')
plt.title('Downsampled MS Image - Band 2')
plt.colorbar(label='Intensity')
plt.xlabel('Column')
plt.ylabel('Row')
plt.show()
plt.savefig('MS_168.png')


# -----------------------------
# 2) Load and downsample hyperspectral ENVI image
# -----------------------------

# Hyperspectral ENVI header (and associated .img)
hdr = io.envi.open('/scratch/r.sugandha/test_deep/guided-deep-decoder/GDD_code/frt00018627_07_if166j_mtr3.hdr')
# Alternative example:
# /scratch/r.sugandha/test_deep/frt000178a3_07_sr168j_mtr3.hdr

hs_image = np.array(hdr.load())   # shape: (H, W, bands)
nan_mask = np.isnan(hs_image)
print("Contains NaN values:", np.any(nan_mask))
print("Number of NaN values:", np.count_nonzero(nan_mask))

print("min HS:", np.min(hs_image))
print("max HS:", np.max(hs_image))

# Check for sentinel 65535
is_65535_present = np.any(hs_image == 65535)
print("65535 present in the data:", is_65535_present)

# Check min/max excluding 65535
valid_mask = hs_image != 65535
min_value = np.min(hs_image[valid_mask])
max_value = np.max(hs_image[valid_mask])
print("Min excluding 65535:", min_value)
print("Max excluding 65535:", max_value)

# Replace 65535 with 0
hs_image[hs_image == 65535] = 0
print("65535 present after replacement:", np.any(hs_image == 65535))
print("HS original shape:", hs_image.shape)

# Downsample HS image in spatial dimensions to 32 x 32
downsampling_factor = (
    32 / hs_image.shape[0],
    32 / hs_image.shape[1],
    1
)
downsampled_hs_image = zoom(hs_image, zoom=downsampling_factor, order=1)
print("downsampled HS shape:", downsampled_hs_image.shape)
print("min downsampled_hs_image:", np.min(downsampled_hs_image))
print("max downsampled_hs_image:", np.max(downsampled_hs_image))

# Visualize a band (e.g., band 120) of the downsampled HS image (optional)
plt.subplot(1, 2, 2)
plt.imshow(
    downsampled_hs_image[:, :, 120],
    cmap='gray',
    vmin=np.percentile(downsampled_hs_image[:, :, 120], 5),
    vmax=np.percentile(downsampled_hs_image[:, :, 120], 95)
)
plt.title('Downsampled Hyperspectral Image (band 120)')
plt.colorbar(label='Intensity')
plt.xlabel('Column')
plt.ylabel('Row')

plt.tight_layout()
plt.show()
plt.savefig('hs120.png')


# -----------------------------
# 3) Load precomputed SRF from MATLAB and save MSI/HSI/R to .mat
# -----------------------------

loaded_data = loadmat('/scratch/r.sugandha/test_deep/SRF_frt00018627_07_if166j_mtr3.mat')
print("Keys in the SRF .mat file:", loaded_data.keys())
R_loaded = loaded_data['srf']

# Drop last column of SRF matrix
SRF_modified = R_loaded[:, :-1]
print("Shape of loaded SRF:", SRF_modified.shape)

# Prepare data for Guided Deep Decoder
data_to_save = {
    'MSI': ctx_image,             # downsampled multispectral (CTX)
    'HSI': downsampled_hs_image,  # downsampled hyperspectral
    'R': SRF_modified             # spectral response function
}

savemat('/scratch/r.sugandha/test_deep/guided-deep-decoder/GDD_code/my_data6.mat', data_to_save)


# -----------------------------
# 4) (Optional) Estimate SRF in Python (instead of using precomputed .mat)
# -----------------------------

def gaussian_downsampling(data, options):
    """
    Gaussian blur + resize to match HSI resolution.
    Used to approximate the MS sensor blur for SRF estimation.
    """
    h_rows, h_cols = options['hRows'], options['hCols']
    p_rows, p_cols = options['pRows'], options['pCols']
    hsize = (round(p_rows / h_rows), round(p_cols / h_cols))
    sigma = hsize[0] / (2 * np.sqrt(2 * np.log(2)))

    data = gaussian_filter(data, sigma=sigma)
    img = Image.fromarray(data)

    # Resize using high-quality downsampling
    resized_img = img.resize((h_cols, h_rows), Image.LANCZOS)
    resized_data = np.array(resized_img)

    return resized_data


def estimate_srf(hcube, pcube, options):
    """
    Estimate spectral response function (SRF) by solving a
    least squares problem between HSI and blurred MS data.
    """
    lr_pan = gaussian_downsampling(pcube, options).reshape(-1, options['pBands'])
    hcube = hcube.reshape(-1, options['hBands'])
    hcube = np.hstack([hcube, np.ones((hcube.shape[0], 1))])  # add bias term
    srf = np.linalg.lstsq(hcube, lr_pan, rcond=None)[0].T
    return srf


# Use downsampled HS and CTX to estimate SRF
hs_cube = downsampled_hs_image
hs_cube[hs_cube == 65535] = 0   # just in case

ms_cube = ctx_image
print("hs cube shape:", hs_cube.shape)
print("ms cube shape:", ms_cube.shape)

parameters = {
    'hRows': hs_cube.shape[0],
    'hCols': hs_cube.shape[1],
    'hBands': hs_cube.shape[2],
    'pRows': ms_cube.shape[0],
    'pCols': ms_cube.shape[1],
    'pBands': ms_cube.shape[2]
}

srf = estimate_srf(hs_cube, ms_cube, parameters)
print("estimated srf shape:", srf.shape)

