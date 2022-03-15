"""
Script to:

1.) extract clouds from pairs of cloudy/cloudless images.
    use flag --extract_clouds
    
2.) Make folder full of cloudless .tif chips from .npz arrays.
    use flag --save_cloudless_as_tif
    
3.) Make cloudbank dataframe to feed to pytorch dataloader for model training.
"""
import numpy as np
import pandas as pd
import pandas_path as path
from pathlib import Path
from PIL import Image
import glob
import argparse
import os

import multiprocessing
import subprocess
import skimage.transform as st

from cloud_seg.utils import utils
from cloud_seg.io import io
from cloud_seg.models.cloudmix import cloud_mlp
from cloud_seg.models.cloudmix import cloud_match

DATA_DIR = Path.cwd().parent.resolve() / "data/"
DATA_DIR_CLOUDS = DATA_DIR / 'clouds/'
DATA_DIR_CLOUDLESS = DATA_DIR / 'cloudless/'
DATA_DIR_CLOUDLESS_MOST_SIMILAR = DATA_DIR / 'cloudless_most_similar/'
DATA_DIR_CLOUDLESS_TIF = DATA_DIR / 'cloudless_tif/'
DATA_DIR_OUT = DATA_DIR / "model_training/"

TRAIN_FEATURES = DATA_DIR / "train_features/"
TRAIN_FEATURES_NEW = DATA_DIR / "train_features_new/"

TRAIN_LABELS = DATA_DIR / "train_labels/"

IMAGE_OUTSIZE = [512, 512]
INTERPOLATION_ORDER = 0

BAD_CHIPS_FILE = DATA_DIR / "BAD_CHIP_DATA/BAD_CHIP_LABEL_IDS.txt"
EASY_CHIPS_FILE = DATA_DIR / "BAD_CHIP_DATA/EASY_CHIPS_IDS.txt"

BAD_CHIP_IDS = list(np.loadtxt(BAD_CHIPS_FILE, dtype=str))
EASY_CHIP_IDS = list(np.loadtxt(EASY_CHIPS_FILE, dtype=str))

assert TRAIN_FEATURES.exists(), TRAIN_LABELS.exists()

Path(DATA_DIR_OUT).mkdir(parents=True, exist_ok=True)
Path(DATA_DIR_CLOUDS).mkdir(parents=True, exist_ok=True)


parser = argparse.ArgumentParser(description='runtime parameters')
parser.add_argument("--bands", nargs='+' , default=["B02", "B03", "B04", "B08"],
                    help="bands desired")

parser.add_argument("--bands_new", nargs='+', default=None,
                    help="additional bands to use beyond original four")

parser.add_argument("-ncv", "--num_cross_validation_splits", type=int, default=5,
                    help="fraction of data to put in validation set") 

parser.add_argument("--save_cloudless_as_tif", action="store_true",
                    help="For each cloudless chip save array of band data (Nimg, H, W) as invididual .tif files") 

parser.add_argument("--extract_clouds", action="store_true",
                    help="Extract clouds from pairs of cloudy and cloudless chips") 

parser.add_argument("--cloud_extract_model", type=str, default='opacity',
                    help="Cloud model to use", choices=['opacity', 'additive']) 

parser.add_argument("--brightness_correct_model", type=str, default='mlp',
                    help="Brightness correcting model to use", choices=[None, 'median', 'mlp']) 

parser.add_argument("--frac_all_cloud_keep", type=float, default=0.1,
                    help="Fraction of total cloud cover cloud chips to keep") 

parser.add_argument("--remake_all", action="store_true",
                    help="Remake all images, and overwrite current ones on disk")

parser.add_argument("--max_pool_size", type=int, default=64,
                help="number of pooling threads to use")

parser.add_argument("--interpolation_order", type=int, default=0,
                    help="interpolation order for resizing images") 

parser.add_argument("--seed", type=int , default=13579,
                    help="random seed for train test split")

parser.add_argument("--dont_save_to_disk", action="store_true",
                    help="save training and validation sets to disk")  

parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")

params = vars(parser.parse_args())
params['bands_use'] = sorted(params['bands'] + params['bands_new']) if params['bands_new'] is not None else params['bands']

params['outsize'] = [512, 512]

if params['verbose']: print("Parameters are: ", params)
    
def construct_cloudbank_dataframe(df_val, params: dict, make_val=False):
    """Construct cloudbank using all chips that do not overlap with validation set"""
    np.random.seed(params['seed'])
    
    cloud_chips = sorted(glob.glob(str(DATA_DIR_CLOUDS) + '/*'))
    
    # Check that files exist in directory
    cloud_chips = [i for i in cloud_chips if os.path.isfile(os.path.join(i,'B04.tif'))] 
    print(f"\nTotal number of cloud chips is {len(cloud_chips)}")

    # remove cloud chips that are from validation sample
    in_val = [os.path.basename(i) in df_val['chip_id'].to_numpy() for i in cloud_chips]
    
    if not make_val:
        cloud_chips = [chip for ichip, chip in enumerate(cloud_chips) if not in_val[ichip]] 
    if make_val:
        cloud_chips = [chip for ichip, chip in enumerate(cloud_chips) if in_val[ichip]] 
        
    print(f"\nTotal number of cloud chips not overlapping validation chips is {len(cloud_chips)}")

    # Get label stats to use to remove certain chips
    labels_mean = np.array([np.mean(np.array(Image.open(os.path.join(chip_dir, 'label.tif')))) for chip_dir in cloud_chips])
    cloud_chips = np.array(cloud_chips)
    
    # Remove cloud chips where mean(label)==0 (these are not cloud chips anyways)
    dm_no_cloud = labels_mean == 0.

    # Subsample chips where mean(label)==1
    dm_all_cloud = labels_mean == 1.
    all_cloud_chips = np.random.choice(
        cloud_chips[dm_all_cloud],
        int(np.sum(dm_all_cloud) * params['frac_all_cloud_keep']),
        replace=False,
    )

    cloud_chips = list(cloud_chips[~dm_no_cloud & ~dm_all_cloud])
    cloud_chips += list(all_cloud_chips)
    
    print(f"\nTotal number of cloud chips after removing selected is {len(cloud_chips)}")

    cloudbank = []
    for chip in cloud_chips:
        # for each location choose an image 
        chip_id = os.path.basename(chip)
        # print(chip, chip_id)

        feature_cols = [chip + f"/{band}.tif" for band in params['bands_use']]
        label_col = [chip + f"/label.tif"]
        # print(feature_cols, label_col)
        cloudbank.append([chip_id]+feature_cols+label_col)

    df_meta = pd.DataFrame(cloudbank, columns=list(df_val.columns)+['label_path'])
    df_meta.head()
    
    # Remove chips with incorrect labels
    print(len(df_meta))
    df_meta = df_meta[~df_meta["chip_id"].isin(BAD_CHIP_IDS)].reset_index(drop=True)
    print(f"\nREMOVING {len(BAD_CHIP_IDS)} BAD LABELS")
    print(len(df_meta))

    return df_meta

def load_validation_dataframe(isplit: int, params: dict):
    
    file_name_in = f"validate_features_meta_cv{isplit}.csv"
    # file_name_in = f"validate_features_meta_seed{params['seed']}_cv{isplit}.csv"

    df_val = pd.read_csv(DATA_DIR_OUT / file_name_in)
          
    return(df_val)

def save_dataframe_to_disk(df_meta, isplit, params: dict, make_val=False):
    
    print(f"\nSaving cloudbank from split {isplit} to disk at:\n{str(DATA_DIR_OUT)}")

    if not make_val:
        file_name_out = f"train_cloudbank_meta_cv{isplit}.csv"

    if make_val:
        file_name_out = f"validate_cloudbank_meta_cv{isplit}.csv"

    df_meta.to_csv(DATA_DIR_OUT / file_name_out, index=False)

def load_npz_arrays_for_chip(chip_id):
    cloudless_chip_dir = DATA_DIR_CLOUDLESS / chip_id
    images_cloudless_all = {}

    for band in params['bands_use']:
        f_ = cloudless_chip_dir / f"{band}.npz" 
        
        d_ = np.load(f_, allow_pickle=True)

        # resize images to outsize
        images_in = np.array(d_["images"]).astype(np.float32)

        single_image = False
        if images_in.shape[0] == 1:
            single_image = True
        if images_in.ndim < 3:
            # only a single image was saved to .npz file
            single_image = True
            images_in = images_in[None, ...]
        
        images_resize = np.zeros((len(images_in), IMAGE_OUTSIZE[0], IMAGE_OUTSIZE[1]), dtype=np.float32)
        for i in range(0, len(images_in)):
            if images_in[i].shape != IMAGE_OUTSIZE:
                images_resize[i] = st.resize(
                    images_in[i].astype(np.float32),
                    IMAGE_OUTSIZE,
                    order=INTERPOLATION_ORDER,
                )
        images_in = images_resize
        
        # images_out = np.zeros( (images_in.shape[0], params['outsize'][0], params['outsize'][1]), dtype=np.float32)
        # for i in range(images_in.shape[0]):
        #     images_out[i] = utils.resize_image(images_in[i], params['outsize'], interpolation_order=params['interpolation_order']) 

        images_cloudless_all[band] = images_in
        if single_image:
            images_cloudless_all[band+"_time"] = [d_["times"]]
            images_cloudless_all[band+"_dtime"] = [d_["dtimes"]]
            images_cloudless_all[band+"_properties"] = [d_["properties"]]

        else:
            images_cloudless_all[band+"_time"] = d_["times"]
            images_cloudless_all[band+"_dtime"] = d_["dtimes"]
            images_cloudless_all[band+"_properties"] = d_["properties"]

        images_cloudless_all[band+"_nimg"] = images_in.shape[0]
        
    nimages = images_cloudless_all["B02_nimg"]
    
    images_matching = np.full(nimages, True, dtype=bool)
    for iimg in range(nimages):
        for iband, band in enumerate(params['bands_use']):
            if iband == 0:
                chip_properties = images_cloudless_all[band+"_properties"][iimg]
            else:
                images_matching[iimg] = images_matching[iimg] and (images_cloudless_all[band+"_properties"][iimg]==chip_properties)
                
    if params['verbose']: print("All images matching? ", images_matching)
     
    images_cloudless_all_matching = {}
    for iband, band in enumerate(params['bands_use']):
        images_cloudless_all_matching[band] =  images_cloudless_all[band][images_matching]
    
    if params['verbose']: print(images_cloudless_all_matching["B02"].shape)
        
    return images_cloudless_all_matching       

def save_npz_chip_arrays_to_tif(cloudless_dir):
    
    chip_id = os.path.basename(cloudless_dir)
    print(chip_id)
    
    try:
        images_cloudless_all = load_npz_arrays_for_chip(chip_id)
    except:
        return
    
    for iimg in range(images_cloudless_all["B02"].shape[0]):
        band_diri = Path(DATA_DIR_CLOUDLESS_TIF / f"{chip_id}/{iimg}/")
        Path(band_diri).mkdir(parents=True, exist_ok=True)

        for band in params['bands_use']:

            band_loc = band_diri / f"{band}.tif"
            # if not os.path.isfile(band_loc):
            band_image = Image.fromarray(images_cloudless_all[band][iimg])        
            band_image.save(band_loc)   

def run_npz_chip_arrays_to_tif():
    
    cloudless_dirs_all = sorted(glob.glob(str(DATA_DIR_CLOUDLESS) + '/*'))
    # check npz exists on disk
    cloudless_dirs_all = [i for i in cloudless_dirs_all if os.path.isfile(os.path.join(i,'B04.npz'))] 

    cloudless_dirs = []

    for ic, cloudless_dir in enumerate(cloudless_dirs_all):

        chip_id = os.path.basename(cloudless_dir)

        # check if output files already exist for this chip. Skip if so
        exists = True 
        if params['remake_all']: 
            exists = False
        
        for band in params['bands_use']:
            if not Path(DATA_DIR_CLOUDLESS_TIF / f'{chip_id}/0/{band}.tif').is_file():
                exists = False
        
        if exists:
            continue
            
        cloudless_dirs.append(cloudless_dir)
    
    if params['max_pool_size'] <= 1:
        for cloudless_dir in cloudless_dirs:
            try:
                save_npz_chip_arrays_to_tif(cloudless_dir)
            except:
                print(f"{cloudless_dir} has no matching")
    else:
        cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpus if cpus < params['max_pool_size'] else params['max_pool_size'])
        print(f"Number of available cpus = {cpus}")

        pool.map(save_npz_chip_arrays_to_tif, cloudless_dirs)#.get()

        pool.close()
        pool.join()  
        
def make_clouds(cloudless_dir):
    
    chip_id = os.path.basename(cloudless_dir)
    print(chip_id)

    # check if output files already exist for this chip. Skip if so
    # if Path(DATA_DIR_CLOUDS / f'{chip_id}/').is_dir() and not params['remake_all']:
    #     if params['verbose']: print(f"{DATA_DIR_CLOUDLESS_TIF / f'{chip_id}/'} already exists")
    #     continue

    image = io.load_image(chip_id, TRAIN_FEATURES, TRAIN_FEATURES_NEW, bands=params['bands_use'])
    label = io.load_label(chip_id, TRAIN_LABELS)

    files = sorted(glob.glob(str(DATA_DIR_CLOUDLESS / chip_id / '*')))
    
    try:
        images_cloudless_all = load_npz_arrays_for_chip(chip_id)
    except:
        return
    
    print("EXTRACTING CLOUDS", chip_id)

    image_cloudless, clouds, opacity_mask = cloud_match.extract_clouds(
        params,
        image,
        label,
        images_cloudless_all,
        cloud_extract_model=params['cloud_extract_model'],
        brightness_correct_model=params['brightness_correct_model'],
    )  

    band_diri = Path(DATA_DIR_CLOUDS / f"{chip_id}")
    Path(band_diri).mkdir(parents=True, exist_ok=True)

    for band in params['bands_use']:

        # save clouds
        band_diri = Path(DATA_DIR_CLOUDS / f"{chip_id}")
        Path(band_diri).mkdir(parents=True, exist_ok=True)

        band_loc = band_diri / f"{band}.tif"
        band_image = Image.fromarray(clouds[band])    
        band_image.save(band_loc)                                  

        # save most similar cloudless image
        band_diri = Path(DATA_DIR_CLOUDLESS_MOST_SIMILAR / f"{chip_id}")
        Path(band_diri).mkdir(parents=True, exist_ok=True)

        band_loc = band_diri / f"{band}.tif"
        band_image = Image.fromarray(image_cloudless[band])        
        band_image.save(band_loc)                                  

    # save label and opacity mask
    band_diri = Path(DATA_DIR_CLOUDS / f"{chip_id}")
    
    band_loc = band_diri / f"label.tif"
    band_labels = Image.fromarray(label)        
    band_labels.save(band_loc)                                  

    band_loc = band_diri / f"opacity.tif"
    band_opacity_mask = Image.fromarray(opacity_mask)        
    band_opacity_mask.save(band_loc)  
    
    return

def run_make_clouds(params: dict):
    
    cloudless_dirs_all = sorted(glob.glob(str(DATA_DIR_CLOUDLESS) + '/*'))[::-1]
    # Check .npz data exists on disk 

    cloudless_dirs = []

    print(params)
    for ic, cloudless_dir in enumerate(cloudless_dirs_all):

        chip_id = os.path.basename(cloudless_dir)

        # check if output files already exist for this chip. Skip if so
        exists = True 
        if params['remake_all']: 
            exists = False

        for band in params['bands_use']:
            if not Path(DATA_DIR_CLOUDS / f'{chip_id}/{band}.tif').is_file():
                exists = False

        if exists:
            continue
        
        cloudless_dirs.append(cloudless_dir)

    if params['max_pool_size'] <= 1:
        for cloudless_dir in cloudless_dirs:
            make_clouds(cloudless_dir)
    else:
        cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpus if cpus < params['max_pool_size'] else params['max_pool_size'])
        print(f"Number of available cpus = {cpus}")

        pool.map(make_clouds, cloudless_dirs)#.get()

        pool.close()
        pool.join()    
        
        
def main():

    if params['save_cloudless_as_tif']:
        run_npz_chip_arrays_to_tif()

    if params['extract_clouds']:
        # Extract all clouds from pairs of cloudy and cloudless chips
        run_make_clouds(params)
    
    for isplit in range(params['num_cross_validation_splits']):
        df_val = load_validation_dataframe(isplit, params)
        
        for make_val in [True, False]:
            df_meta = construct_cloudbank_dataframe(df_val, params, make_val=make_val)

            if not params['dont_save_to_disk']:
                save_dataframe_to_disk(df_meta, isplit, params, make_val=make_val)
  
if __name__=="__main__":
    main()
