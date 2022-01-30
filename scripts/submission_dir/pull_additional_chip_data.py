"""Pull additional data from Microsoft's Planetary computer"""
import numpy as np
import pandas as pd
import skimage.transform as st

import rioxarray
import rasterio
import rioxarray

import sys
import os
import time
from PIL import Image
from pathlib import Path
from pandas_path import path
from tqdm import tqdm

import multiprocessing
import subprocess
# from contextlib import redirect_stdout

import argparse

try:
    from cloud_seg.pc_apis import query_bands
    from cloud_seg.utils import utils
except:
    import query_bands
    import utils
    
from datetime import date

if os.path.isdir('../data/train_features'):
    print("Running locally")
    label_str = 'train'
    
    DATA_DIR = Path.cwd().parent.resolve() / "data/"
    DATA_DIR_OUT = Path.cwd().parent.resolve() / "data/"

else:
    print("Not running locally")
    # not running locally - assume on planetary computer
    label_str = 'test'
    
    # locations to various directories
    DATA_DIR = "data/"
    DATA_DIR_OUT = "data_new/"
    Path(DATA_DIR_OUT).mkdir(parents=True, exist_ok=True)

    BANDS_NEW = ['B01', 'B11']
    
FEATURES = DATA_DIR / "{:s}_features".format(label_str)
LABELS   = None
METADATA = DATA_DIR / "{:s}_metadata.csv".format(label_str)

IMAGE_OUTSIZE = [512, 512]
INTERPOLATION_ORDER = 0

def parse_commandline_arguments() -> "argparse.Namespace":
    """Parse commandline arguments."""

    parser = argparse.ArgumentParser(description='runtime parameters')
    parser.add_argument("--max_pool_size", type=int, default=64,
                        help="number of pooling threads to use")
    
    parser.add_argument("--new_bands", nargs='+' , default=["B02", "B03", "B04", "B08"],
                        help="bands desired")
    
    parser.add_argument("--new_band_dirs", nargs='+', default=[],
                        help="directories to save bands in")
    
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    
    parser.add_argument("--collection", type=str, default="sentinel-2-l2a",
                        help="planetary collection to search")
    
    parser.add_argument("--query_range_minutes", type=float, default=60 * 24 * 365 * 5,
                        help="time range from original chip to query")
    
    parser.add_argument("--want_closest", action="store_true",
                        help="If true return only return closest chip to query (possibly query chip itself). \
                        Else return closest non-matching chips")
    
    parser.add_argument("--new_location", action="store_true",
                        help="Pull data from new location")
    
    parser.add_argument("--only_cloudless", action="store_true",
                        help="If true return only return closest chip to query (possibly query chip itself). \
                        Else return closest non-matching chips")
    
    parser.add_argument("--max_cloud_cover", type=float, default=1.0,
                        help="only return chips with below this cloud cover")
    
    parser.add_argument("--max_cloud_shadow_cover", type=float, default=1.0,
                        help="only return chips with below this cloud shadow cover")
    
    parser.add_argument("--max_item_limit", type=int, default=5,
                        help="Maximum number of nearest items to return")

    parser.add_argument("--remake_all", action="store_true",
                        help="Remake all images, and overwrite current ones on disk") 

    args = parser.parse_args()
    return args

#     parser.add_argument("", type=int, default=,
#                         help="")


### Load params and data at top of script, and not in main(), 
### as multiprocessing.pool does not like dictionary argments passed to map function

# load the provided metadata
df = pd.read_csv(METADATA)

# add existing bands
df = utils.add_paths(df, FEATURES, LABELS)

if os.path.isdir('../data/train_features'):
    try:
        lat, lon = np.loadtxt("../data/interesting_places/lat_long.txt", unpack=True, delimiter=',')
        dlat = 0.5
        dlon = 0.5
        #left bottom right left
        lat_lon = np.c_[lon-dlon/2, lat-dlat/2, lon+dlon/2, lat+dlat/2]
        df_lat_lon = pd.DataFrame(lat_lon, columns=['left', 'bottom', 'right', 'top'])
    except:
        print("No new interesting_places file to search for")
        
    params = vars(parse_commandline_arguments())
    
    if params['want_closest'] and not params['new_location']:
        DATA_DIR_OUT = DATA_DIR_OUT / "train_features_new/"
    elif params['new_location']:
        DATA_DIR_OUT = DATA_DIR_OUT / "cloudless_newlocations/"
    else:
        DATA_DIR_OUT = DATA_DIR_OUT / "cloudless/"

else:
    print("not running locally")
    params = vars(parse_commandline_arguments())
    params['want_closest'] = True
    params['query_range_minutes'] = 180
    DATA_DIR_OUT = DATA_DIR_OUT / "test_features_new/"

    params['new_bands'] = BANDS_NEW
    params['max_pool_size'] = 8
    
if params['new_band_dirs'] == []:
    # no specific output directories specified, default to band names
    params['new_band_dirs'] = params['new_bands']

Path(DATA_DIR_OUT).mkdir(parents=True, exist_ok=True)
params['DATA_DIR_OUT'] = DATA_DIR_OUT
print(params)

def download_assets(irow):
    
    row = df.iloc[irow]
    
    pystac_chip = PystacAsset(params, df_chip=row)
    
    if not pystac_chip.exists_on_disk:
        tstart = time.time()
        
        pystac_chip.get_assets_from_chip()
        
        pystac_chip.save_assets_to_disk()
        
        tend = time.time()
        if irow % 10 == 0:
            
            print("Download time for chip was {:.03f} s".format(tend-tstart))
                
def download_location_assets(irow):
                
    lat_lon = df_lat_lon.iloc[irow]
    lat_lon['crs'] = 'EPSG:4326' #'EPSG:32736'
    lat_lon['datetime'] = '2020-04-29T08:20:47Z'
    lat_lon['chip_id'] = index_to_chip_string(irow, nchars_per_string=4)
    
    pystac_chip = PystacAsset(params, lat_lon=lat_lon)

    if not pystac_chip.exists_on_disk:
        tstart = time.time()
        
        pystac_chip.get_assets_from_chip()
        
        try:
            pystac_chip.save_assets_to_disk()
        except:
            print("no chips to save")
        tend = time.time()
        if irow % 10 == 0:
            print("Download time for chip was {:.03f} s".format(tend-tstart))

def index_to_chip_string(index, nchars_per_string=4):
    """create string AAAA, AAAB, ... for given index in 0,1,..."""
    nchars_in_alphabet = 26

    ind_char_start = 65
    ind_chat_end = 91

    chip_string = ''
    for i in range(nchars_per_string-1, -1, -1):
        chip_string += chr(ind_char_start+(index//nchars_in_alphabet**i)%nchars_in_alphabet)
    
    return chip_string
    
class PystacAsset:
    def __init__(self, params, df_chip=None, lat_lon=None):

        self.df_chip = df_chip
        if df_chip is not None:
            # load locations from alteady existing geotiff
            self.chip_id = df_chip.chip_id
        elif lat_lon is not None:
            # load locations from lat_lon
            self.lat_lon = lat_lon
            self.datetime = lat_lon['datetime'] 
            self.chip_id = lat_lon['chip_id']
        else:
            sys.exit("no chip or lat_lon provided")
            
        self.verbose = params.get("verbose", True)
        self.collection = params.get("collection", "sentinel-2-l2a")
        
        self.new_bands = params.get("new_bands", ["B02", "B03", "B04", "B08"])
        self.new_band_dirs = params.get("new_band_dirs", ["B02", "B03", "B04", "B08"])
        self.DATA_DIR_OUT = params.get("DATA_DIR_OUT", "data/cloudless_test")
        self.remake_all = params.get("remake_all", False)
        
        self.query_range_minutes = params.get("query_range_minutes", 60 * 24 * 365 * 5)
        self.want_closest = params.get("want_closest", False)
        self.max_cloud_cover = params.get("max_cloud_cover", 1.0)
        self.max_cloud_shadow_cover = params.get("max_cloud_shadow_cover", 1.0)
        self.max_item_limit = params.get("max_item_limit", 5)

        self.file_extension = '.npz'
        if self.want_closest:
            self.file_extension = '.tif'
            
        self.exists_on_disk = self.check_if_bands_on_disk()
        
        self.bad_chip = False
        
    def check_if_bands_on_disk(self):
        """check if all desired new data already exists for this chip"""
        exists_on_disk=False
        if not self.remake_all:
            exists_on_disk = True

            current_band_dir = os.path.join(self.DATA_DIR_OUT, f"{self.chip_id}")
            for band, band_dir in zip(self.new_bands, self.new_band_dirs):
    
                if not os.path.isfile(os.path.join(current_band_dir, f"{band_dir}{self.file_extension}")):
                    exists_on_disk = False

        return exists_on_disk

    def resize_images(self, images):
        """resize all images to size of first in list"""
        image_shapes = np.unique([i.shape for i in images])
        image_shape_nearest = images[0].shape
        image_dtype = images[0].dtype

        if images.ndim < 3:
            # only a single image was saved to .npz file
            single_image = True
            images = images[None, ...]

        for i in range(0, len(images)):
            if images[i].shape != IMAGE_OUTSIZE:
                images[i] = st.resize(
                    images[i].astype(np.float32),
                    IMAGE_OUTSIZE,
                    order=INTERPOLATION_ORDER,
                )
            images[i] = images[i].astype(image_dtype)

        return images
                
    def save_assets_to_disk(self):
        
        if self.verbose: print('Saving to disk')

        for band, band_dir in zip(self.new_bands, self.new_band_dirs):
            
            if self.want_closest:
                """
                Save each band as .tif.
                Useful for pulling additional band features corresponding to train/test chips
                """
                if self.df_chip is not None:
                    try:
                        # resize image to 512, 512
                        self.assets[band] = st.resize(
                            self.assets[band].astype(np.float32),
                            IMAGE_OUTSIZE,
                            order=INTERPOLATION_ORDER,
                        )
                    except:
                        self.assets = {}
                        print(f"Band {band} was not found or weird shape. Saving zeros instead")
                        self.assets[band] = np.full(IMAGE_OUTSIZE, 0, dtype=np.uint8)

                band_diri = Path(self.DATA_DIR_OUT / f"{self.chip_id}")
                band_diri.mkdir(parents=True, exist_ok=True)
                
                band_image = Image.fromarray(self.assets[band])
                band_image.save(band_diri / f"{band_dir}.tif")  

            else:
                """
                Save multiple bands in .npz
                Used mostly for pulling cloudless versions of chips in dataset 
                """
                band_diri = Path(self.DATA_DIR_OUT / f"{self.chip_id}")
                Path(band_diri).mkdir(parents=True, exist_ok=True)
                
                try:
                    self.assets[band] = self.resize_images(self.assets[band])

                    np.savez(
                        band_diri / f"{band_dir}.npz",
                        images=np.stack(self.assets[band], axis=0),
                        times=self.assets[band + "_time"],
                        dtimes=self.assets[band + "_dtime"],
                        properties=self.assets[band + "_properties"],
                    )
                except:
                    print(f"{band_diri} has no chips")
                
    def get_assets_from_chip(self):

        tstart = time.time()
        # Load extra bands from PySTAC
        # self.assets, self.items = query_bands.query_bands_from_lat_lon(
        #     self.lat_lon,
        #     timestamp=self.datetime,
        #     asset_keys=self.new_bands,
        #     collection=self.collection,
        #     query_range_minutes=self.query_range_minutes,
        #     verbose=self.verbose,
        #     want_closest=self.want_closest,
        #     max_cloud_cover=self.max_cloud_cover,
        #     max_cloud_shadow_cover=self.max_cloud_shadow_cover,
        #     max_item_limit=self.max_item_limit,
        # )                  
        try:
            if self.df_chip is not None:
                # Load extra bands from PySTAC
                self.assets, self.items = query_bands.query_bands(
                    rasterio.open(self.df_chip.B04_path),
                    timestamp=self.df_chip.datetime,
                    asset_keys=self.new_bands,
                    collection=self.collection,
                    query_range_minutes=self.query_range_minutes,
                    verbose=self.verbose,
                    want_closest=self.want_closest,
                    max_cloud_cover=self.max_cloud_cover,
                    max_cloud_shadow_cover=self.max_cloud_shadow_cover,
                    max_item_limit=self.max_item_limit,
                )
            if self.lat_lon is not None:
                # Load extra bands from PySTAC
                self.assets, self.items = query_bands.query_bands_from_lat_lon(
                    self.lat_lon,
                    timestamp=self.datetime,
                    asset_keys=self.new_bands,
                    collection=self.collection,
                    query_range_minutes=self.query_range_minutes,
                    verbose=self.verbose,
                    want_closest=self.want_closest,
                    max_cloud_cover=self.max_cloud_cover,
                    max_cloud_shadow_cover=self.max_cloud_shadow_cover,
                    max_item_limit=self.max_item_limit,
                )                
        except:
            print(f"{self.chip_id} sucks and we can't find it")
            self.bad_chip = True
            
        if self.verbose: print('Got assets')


def main():

    if not params['new_location']:
        if params['max_pool_size'] <= 1:
            for i in range(len(df)):
                download_assets(i)
        else:
            cpus = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(cpus if cpus < params['max_pool_size'] else params['max_pool_size'])
            print(f"Number of available cpus = {cpus}")

            pool.map(download_assets, range(len(df)))#.get()

            pool.close()
            pool.join()

    else:
        if params['max_pool_size'] <= 1:
            for i in range(len(df_lat_lon)):
                download_location_assets(i)
        else:
            cpus = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(cpus if cpus < params['max_pool_size'] else params['max_pool_size'])
            print(f"Number of available cpus = {cpus}")

            pool.map(download_location_assets, range(len(df_lat_lon)))#.get()

            pool.close()
            pool.join()
            
if __name__=="__main__":

    main()
