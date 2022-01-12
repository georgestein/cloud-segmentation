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

from cloud_seg.utils import utils
from cloud_seg.io import io

DATA_DIR = Path.cwd().parent.resolve() / "data/"
DATA_DIR_CLOUDS = DATA_DIR / 'clouds/'
DATA_DIR_CLOUDLESS = DATA_DIR / 'cloudless/'
DATA_DIR_CLOUDLESS_MOST_SIMILAR = DATA_DIR / 'cloudless_most_similar/'
DATA_DIR_CLOUDLESS_TIF = DATA_DIR / 'cloudless_tif/'
DATA_DIR_OUT = DATA_DIR / "model_training/"

TRAIN_FEATURES = DATA_DIR / "train_features/"
TRAIN_FEATURES_NEW = DATA_DIR / "train_features_new/"

TRAIN_LABELS = DATA_DIR / "train_labels/"

assert TRAIN_FEATURES.exists(), TRAIN_LABELS.exists()

Path(DATA_DIR_OUT).mkdir(parents=True, exist_ok=True)
Path(DATA_DIR_CLOUDS).mkdir(parents=True, exist_ok=True)

def construct_cloudbank_dataframe(df_val, params: dict):
    """Construct cloudbank using all chips that do not overlap with validation set"""
    cloud_chips = sorted(glob.glob(str(DATA_DIR_CLOUDS) + '/*'))
    
    print(f"\nTotal number of cloud chips is {len(cloud_chips)}")
    
    # remove cloud chips that are from validation sample
    in_val = [os.path.basename(i) in df_val['chip_id'].to_numpy() for i in cloud_chips]
    cloud_chips = [chip for ichip, chip in enumerate(cloud_chips) if not in_val[ichip]] 

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
    print(f"Size of cloudbank not overlapping validation chips is {len(df_meta)} chips")
    df_meta.head()
    
    return df_meta

def load_validation_dataframe(isplit: int, params: dict):
    
    file_name_in = f"validate_features_meta_cv{isplit}.csv"
    # file_name_in = f"validate_features_meta_seed{params['seed']}_cv{isplit}.csv"

    df_val = pd.read_csv(DATA_DIR_OUT / file_name_in)
          
    return(df_val)

def save_dataframe_to_disk(df_meta, isplit, params: dict):
    
    print(f"\nSaving cloudbank from split {isplit} to disk at:\n{str(DATA_DIR_OUT)}")

    file_name_out = f"cloudbank_meta_cv{isplit}.csv"
    # file_name_out = f"cloudbank_meta_seed{params['seed']}_cv{isplit}.csv"

    df_meta.to_csv(DATA_DIR_OUT / file_name_out, index=False)

def load_npz_arrays_for_chip(params, chip_id):
    
    cloudless_chip_dir = DATA_DIR_CLOUDLESS / chip_id
    images_cloudless_all = {}

    for band in params['bands_use']:
        f_ = cloudless_chip_dir / f"{band}.npz" 
        
        d_ = np.load(f_, allow_pickle=True)

        # resize images to outsize
        images_in = np.array(d_["images"]).astype(np.float32)

        single_image = False
        if images_in.ndim < 3:
            # only a single image was saved to .npz file
            single_image = True
            images_in = images_in[None, ...]

            
        images_out = np.zeros( (images_in.shape[0], params['outsize'][0], params['outsize'][1]), dtype=np.float32)
        for i in range(images_in.shape[0]):
            images_out[i] = utils.resize_image(images_in[i], params['outsize'], interpolation_order=params['interpolation_order']) 

        images_cloudless_all[band] = images_out
        if single_image:
            images_cloudless_all[band+"_time"] = [d_["times"]]
            images_cloudless_all[band+"_dtime"] = [d_["dtimes"]]
            images_cloudless_all[band+"_properties"] = [d_["properties"]]

        else:
            images_cloudless_all[band+"_time"] = d_["times"]
            images_cloudless_all[band+"_dtime"] = d_["dtimes"]
            images_cloudless_all[band+"_properties"] = d_["properties"]

        images_cloudless_all[band+"_nimg"] = images_out.shape[0]
        
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

def save_npz_chip_arrays_to_tif(params: dict):
    
    cloudless_dirs = sorted(glob.glob(str(DATA_DIR_CLOUDLESS) + '/*'))

    for ic, cloudless_dir in enumerate(cloudless_dirs):

        if ic % 100 == 0:
            print('Running on ', ic)
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
        images_cloudless_all = load_npz_arrays_for_chip(params, chip_id)
    
        for iimg in range(images_cloudless_all["B02"].shape[0]):
            band_diri = Path(DATA_DIR_CLOUDLESS_TIF / f"{chip_id}/{iimg}/")
            Path(band_diri).mkdir(parents=True, exist_ok=True)

            for band in params['bands_use']:

                band_loc = band_diri / f"{band}.tif"
                # if not os.path.isfile(band_loc):
                band_image = Image.fromarray(images_cloudless_all[band][iimg])        
                band_image.save(band_loc)   

def find_and_return_most_similar_image(params, image, label, images_cloudless, brightness_correct_image_cloudless=False):
    """Given a cloudy chip and a number of cloudless versions of the same area, choose or create 
    the most similar one to the cloudy chip.
    
    The simple approximation is to just calculate which set of images best matches in regions where labels==0. 
    This does not accound for shadows.
    """
    
     # determine which new cloudless image is most similar to the old
     # by calculating agreement in non-cloudy regions
    diffs = np.zeros( (len(params['bands_use']), images_cloudless['B02'].shape[0]) )
    for i, band in enumerate(params['bands_use']):

        diff = (image[band]-images_cloudless[band]) * label
        diffs[i] = np.sum(diff, axis=(1,2))

        if diffs[i].max() > 0.:
            # if totally cloud covered label==0 everywhere, and max will be 0.
            diffs[i] /= diffs[i].max()

    total_diffs = np.mean(diffs, axis=0)

    ind_min_band_diff = np.argmin(total_diffs)  
    
    image_cloudless = {}
    for band in params['bands_use']:
        image_cloudless[band] = images_cloudless[band][ind_min_band_diff]
    
        if brightness_correct_image_cloudless:
            # try to match the average intensity in non cloudy regions
            dm = label == 1

            if np.sum(dm) > 0:
                mean_diff = np.median(image[band][dm] - image_cloudless[band][dm])
            else:
                mean_diff = 1.

            # print('mean_diff', mean_diff)
            images_cloudless[band] += mean_diff

    return image_cloudless

def extract_clouds(params, image, label, images_cloudless, cloud_extract_model='additive'):
    """Given cloudy image/label pair, and 'cloudless' images of the same area pulled from the planetary computer,
    extract brightness changes due to clouds.
    
    The simplest model is to assume clouds simply add brightness to each pixel that they cover. 
    If true, assuming that the land does not change between when the cloudy and cloudless images were taken,
    clouds = (images - images_cloudless)*labels.
    
    Unfortunately, both of these assumptions are incorrect
    
    1.) The cloudy and cloudless images are of the same location, but are often seperated by months or years.
        Over this timeframe plants change color, water levels change, and human infractstructure near cities changes.
        Additionally, the images might not be taken from the same angle, causing mis-alignments between each image set.
        
    2.) Clouds are sometimes transparent, sometimes not. An additive model does not correcely account for this
    
    3.) Cloud shadows... We know what angle the sun makes for each chip (in chip properties) can we come up with a way to project these?
    
    """
    cloud_extract_models = ['additive'] # Add transparency later
    
    if cloud_extract_model not in cloud_extract_models:
        print(f"WARNING: cloud model {cloud_extract_model} is not a possible value to use. Using {cloud_extract_models[0]} instead \
            Possible choices are:", cloud_extract_models)
        
    image_cloudless = find_and_return_most_similar_image(params, image, label, images_cloudless)   
    
    # and save to disk as .tif
    clouds = {}
    for band in params['bands_use']:

        if cloud_extract_model=='additive':
            clouds[band] = (image[band] - image_cloudless[band]) 

    return image_cloudless, clouds

def make_clouds(params, cloudless_dir):
    
    chip_id = os.path.basename(cloudless_dir)
    print(chip_id)

    # check if output files already exist for this chip. Skip if so
    # if Path(DATA_DIR_CLOUDS / f'{chip_id}/').is_dir() and not params['remake_all']:
    #     if params['verbose']: print(f"{DATA_DIR_CLOUDLESS_TIF / f'{chip_id}/'} already exists")
    #     continue

    image = io.load_image(chip_id, TRAIN_FEATURES, TRAIN_FEATURES_NEW, bands=params['bands_use'])
    label = io.load_label(chip_id, TRAIN_LABELS)

    files = sorted(glob.glob(str(DATA_DIR_CLOUDLESS / chip_id / '*')))

    images_cloudless_all = load_npz_arrays_for_chip(params, chip_id)

    image_cloudless, clouds = extract_clouds(params, image, label, images_cloudless_all, cloud_extract_model='additive')  

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

    # save label
    band_diri = Path(DATA_DIR_CLOUDS / f"{chip_id}")

    band_loc = band_diri / f"label.tif"
    band_labels = Image.fromarray(label)        
    band_labels.save(band_loc)                                  

def run_make_clouds(params: dict):
    
    cloudless_dirs = sorted(glob.glob(str(DATA_DIR_CLOUDLESS) + '/*'))

    for ic, cloudless_dir in enumerate(cloudless_dirs):

        if ic % 10 == 0:
            print('Running on ', ic)
 
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
           
        make_clouds(params, cloudless_dir)
        
def main():
    
    parser = argparse.ArgumentParser(description='runtime parameters')
    parser.add_argument("--bands", nargs='+' , default=["B02", "B03", "B04", "B08"],
                        help="bands desired")
    parser.add_argument("--bands_new", nargs='+', default=None,
                        help="additional bands to use beyond original four")
    parser.add_argument("-ncv", "--num_cross_validation_splits", type=int, default=4,
                        help="fraction of data to put in validation set") 
    parser.add_argument("--save_cloudless_as_tif", action="store_true",
                        help="For each cloudless chip save array of band data (Nimg, H, W) as invididual .tif files") 
        
    parser.add_argument("--extract_clouds", action="store_true",
                        help="Extract clouds from pairs of cloudy and cloudless chips") 
    
    parser.add_argument("--remake_all", action="store_true",
                        help="Remake all images, and overwrite current ones on disk") 
    
    parser.add_argument("--interpolation_order", type=int, default=1,
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
    
    if params['save_cloudless_as_tif']:
        save_npz_chip_arrays_to_tif(params)

    if params['extract_clouds']:
        # Extract all clouds from pairs of cloudy and cloudless chips
        run_make_clouds(params)
    
    for isplit in range(params['num_cross_validation_splits']):
        df_val = load_validation_dataframe(isplit, params)
        
        df_meta = construct_cloudbank_dataframe(df_val, params)

        if not params['dont_save_to_disk']:
            save_dataframe_to_disk(df_meta, isplit, params)
  
if __name__=="__main__":
    main()
