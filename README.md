# Guided Deep Decoder for Hyperspectral–Multispectral Fusion

This repository contains my course project adapting **Guided Deep Decoder (GDD)** to fuse **hyperspectral (HSI)** and **multispectral (MSI)** planetary data (e.g., CRISM + CTX).

The goal is to:

- Align the multispectral image (CTX) to the hyperspectral footprint,
- Prepare paired HSI/MSI cubes,
- Obtain a **spectral response function (SRF)** that links HSI bands to MSI bands,
- Save everything into a MATLAB `.mat` file (`MSI`, `HSI`, `R`) that can be used as input to the original Guided Deep Decoder code.

> **Note:** This repo contains only *my own* preprocessing / data-preparation code.  
> The actual GDD model and training loop come from the original authors’ implementation (linked below).

---

## Reference Method and Original Code

This work is based on:

> T. Uezato, D. Hong, N. Yokoya, W. He.  
> *Guided Deep Decoder: Unsupervised Image Pair Fusion*, ECCV 2020.

Original PyTorch implementation by the authors:  
https://github.com/tuezato/guided-deep-decoder

If you use the GDD method, please cite their paper.

---

## Repository Contents

The main scripts are:

### `crop_ms_to_hs_extent.py`

- Reads a CRISM-style metadata text file (e.g. `frt00018627_07_if166j_mtr3.txt`) to extract:
  - `MAXIMUM_LATITUDE`, `MINIMUM_LATITUDE`
  - `WESTERNMOST_LONGITUDE`, `EASTERNMOST_LONGITUDE`
- Uses `rasterio` to crop a **georeferenced multispectral image** (e.g., CTX GeoTIFF) to match the hyperspectral footprint.

This is used when you start from large CTX images and want to clip them to the area covered by CRISM.

---

### `prepare_hs_ms_srf_mat.py`

Works directly from the **original sensor files**:

- Loads a CTX **GeoTIFF** with `rasterio` and downsamples it to a fixed spatial size (e.g., 1024 × 1024) → `ctx_image`.
- Loads a CRISM **ENVI hyperspectral cube** (`.hdr` + `.img`) with `spectral.io.envi`, replaces invalid values (65535 → 0) and downsamples it spatially to a small grid (e.g., 32 × 32) → `downsampled_hs_image`.
- Loads a **precomputed SRF** from a MATLAB `.mat` file (e.g., `SRF_frt00018627_07_if166j_mtr3.mat`), drops the last column if needed, and stores it as `R`.
- Saves a `.mat` file for GDD, e.g.:

  - `MSI` – the downsampled CTX multispectral cube  
  - `HSI` – the downsampled CRISM hyperspectral cube  
  - `R` – the spectral response function from the MATLAB SRF file  

The script also includes an optional Python routine (`gaussian_downsampling` + `estimate_srf`) to **estimate SRF directly from HSI+MSI** via Gaussian blur + least squares. In the current version, this estimated SRF is printed and used for debugging/comparison, while the saved `R` comes from the MATLAB SRF file.

---

### `estimate_srf_and_prepare_from_registered_mat.py`

Works from **already registered `.mat` files** instead of raw ENVI/TIFF:

- Loads:
  - Registered hyperspectral cube (`imgHS_cropped`)  
   
  - Registered multispectral cube (`imgMS_cat`) 
- Downsamples the registered hyperspectral cube spatially (e.g., to 32 × 32).
- Estimates the **spectral response function (SRF)** between HSI and MSI by:
  - Blurring the MSI with a Gaussian and resizing it to the HSI resolution,
  - Solving a linear least squares problem to find an SRF matrix `R` such that  
    `HSI × Rᵀ ≈ blurred_MSI`.
- Saves a `.mat` file for GDD, e.g.:

  - `MSI` – registered multispectral cube  
  - `HSI` – downsampled registered hyperspectral cube  
  - `R` – SRF estimated in Python  

This script provides an alternative pipeline when the registration is done externally (e.g., in MATLAB) and you want a fully Python-based SRF estimate.

---

## Typical Workflow (High-Level)

1. **(Optional) Crop CTX to CRISM footprint**

   Use `crop_ms_to_hs_extent.py` if your CTX image is much larger than the CRISM footprint:

   - Input: CRISM metadata `.txt`, CTX GeoTIFF  
   - Output: cropped CTX GeoTIFF matching CRISM lat/lon extent

2. **Prepare GDD input from raw sensor files**

   Use `prepare_hs_ms_srf_mat.py` when you have:

   - CTX GeoTIFF  
   - CRISM ENVI (`.hdr` + `.img`)  
   - Precomputed SRF `.mat`

   This creates a `.mat` file with `MSI`, `HSI`, and `R` that can be used directly with the original GDD code.

3. **Alternative: prepare GDD input from registered `.mat` files**

   Use `process.py` when you already have **registered CRISM/CTX cubes in MATLAB format** and want to:

   - Downsample HSI
   - Estimate SRF in Python
   - Save a `.mat` file with `MSI`, `HSI`, `R` for GDD

4. **Run Guided Deep Decoder**

   - Clone the original GDD repo:

     ```bash
     git clone https://github.com/tuezato/guided-deep-decoder.git
     ```

   - follow the authors’ instructions to run fusion.

---

## Requirements

Main Python dependencies (versions are approximate; exact versions depend on your environment):

- `numpy`
- `scipy`
- `matplotlib`
- `rasterio`
- `spectral`
- `tifffile`
- `Pillow`

You can install these via:

```bash
pip install numpy scipy matplotlib rasterio spectral tifffile pillow




## Using This with the Original GDD Code

1. Clone the original Guided Deep Decoder repo from the authors:

   ```bash
   git clone https://github.com/tuezato/guided-deep-decoder.git GDD_code
