import rasterio
from rasterio.windows import from_bounds

def extract_coords_from_file(file_path):
    """
    Extract geographic bounds (lat/lon) from a CRISM-style metadata text file.

    Looks for the following keys:
    - MAXIMUM_LATITUDE
    - MINIMUM_LATITUDE
    - WESTERNMOST_LONGITUDE
    - EASTERNMOST_LONGITUDE
    """
    bounds = {}
    with open(file_path, 'r') as f:
        for line in f:
            if 'MAXIMUM_LATITUDE' in line:
                bounds['Maximum Latitude'] = float(line.split('=')[1].strip().split('<')[0].strip())
            elif 'MINIMUM_LATITUDE' in line:
                bounds['Minimum Latitude'] = float(line.split('=')[1].strip().split('<')[0].strip())
            elif 'WESTERNMOST_LONGITUDE' in line:
                bounds['Westernmost Longitude'] = float(line.split('=')[1].strip().split('<')[0].strip())
            elif 'EASTERNMOST_LONGITUDE' in line:
                bounds['Easternmost Longitude'] = float(line.split('=')[1].strip().split('<')[0].strip())
    return bounds

def crop_image_to_extent(image_path, output_path, bounds):
    """
    Crop a georeferenced image (e.g., CTX GeoTIFF) to the given geographic extent.

    Parameters
    ----------
    image_path : str
        Path to the input image (GeoTIFF).
    output_path : str
        Path where the cropped GeoTIFF will be written.
    bounds : dict
        Dictionary with keys:
        - 'Westernmost Longitude'
        - 'Easternmost Longitude'
        - 'Minimum Latitude'
        - 'Maximum Latitude'
    """
    with rasterio.open(image_path) as src:
        west = bounds['Westernmost Longitude']
        east = bounds['Easternmost Longitude']
        south = bounds['Minimum Latitude']
        north = bounds['Maximum Latitude']

        window = from_bounds(west, south, east, north, src.transform)
        cropped_image = src.read(window=window)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": window.height,
            "width": window.width,
            "transform": rasterio.windows.transform(window, src.transform)
        })

        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(cropped_image)

def main():
    # Example paths (replace with your own)
    metadata_file_path = '/path/to/frt00018627_07_if166j_mtr3.txt'
    input_ms_image_path = '/path/to/F21_044125_2056_XN_25N021W.tiff'
    output_cropped_path = '/path/to/F21_044125_2056_XN_25N021W_cropped.tiff'

    bounds = extract_coords_from_file(metadata_file_path)
    print("Extracted bounds:", bounds)

    crop_image_to_extent(input_ms_image_path, output_cropped_path, bounds)
    print("Cropped image written to:", output_cropped_path)

if __name__ == "__main__":
    main()

