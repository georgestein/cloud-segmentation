from cloud_seg.io.features_from_sampled_datasets import create_compiled_dataset, create_chip_mask

bad_chip_path='/pscratch/sd/g/gstein/machine_learning/cloud-segmentation/data/BAD_CHIP_DATA/BAD_CHIP_LABEL_IDS.txt'
bands = ['B02', 'B03', 'B04', 'B08',
         'B05', 'B06', 'B07', 'B09',
         'B8A', 'B11', 'B12', 'B01', 
         'SCL', 'LC']


create_compiled_dataset([bands[0]], 'unetpreds', sample_by_LC=False, smooth_sigma=None)
create_compiled_dataset([bands[0]], 'unetpreds_LC', sample_by_LC=True, smooth_sigma=None)
