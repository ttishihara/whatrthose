# FRCNN Prediction 

## Getting Started

This folder is structured as follows
 - `/output_images`: Output of `predict.py` 
 - `/weights`: Model weights are stored here. 
 - `/input_images`: Input images are stored here.  

## Prediction 

To predict images from `/input_images` folder run:

```
python predict.py
```

This saves the the cropped images in the `output_images` folder. 


Currently, the following paths are hardcoded, but can be added as command line arguments in the future: 

- `test_path` - /input_images
- `weight_path` - /weights/model_frcnn_vgg.hdf5
- `config_path` - /weights/model_vgg/config.pickle
- `save_img_path` - /output_images
