import numpy as np

def true_color_band(band_data, nodata=1, pixel_max=255, c=10., th=0.125):
    """
    Normalize band with:    
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    range_val = max_val - min_val
    
    out = (data.astype(np.float32) - min_val)/range_val
    out = 1. / (1. + np.exp(c * (th - out)))
    
    Copied from https://xarray-spatial.org/_modules/xrspatial/multispectral.html#true_color
    """
    # a = np.where(np.logical_or(np.isnan(r), r <= nodata), 0, 255)
    pixel_max = 255

    # h, w = band_data.shape
    # out = np.zeros((h, w, 4), dtype=np.uint8)
    # out[:, :, 0] = (normalize_data_numpy(r, pixel_max, c, th)).astype(np.uint8)
    # out[:, :, 1] = (normalize_data_numpy(g, pixel_max, c, th)).astype(np.uint8)
    # out[:, :, 2] = (normalize_data_numpy(b, pixel_max, c, th)).astype(np.uint8)
    
    out = normalize_data_xrspatial(band_data, pixel_max, c, th) #).astype(np.uint8)

    return out

def normalize_data_xrspatial(data, pixel_max, c, th):
    """
    Copied from https://xarray-spatial.org/_modules/xrspatial/multispectral.html#true_color
    """
    #min_val = np.nanmin(data)
    #max_val = np.nanmax(data)
    min_val = 100
    max_val = 10000
    data = np.clip(data, min_val, max_val)
    
    range_val = max_val - min_val
    
    out = (data.astype(np.float32) - min_val)/range_val
    out = 1. / (1. + np.exp(c * (th - out)))
    
    return out * pixel_max
