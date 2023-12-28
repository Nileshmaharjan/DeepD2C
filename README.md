

## Installation
- Install tensorflow (tested with tf 1.13)
- Python 3 required
- Download dependencies
```bash=
pip install -r requirements.txt
```

## Training
### Encoder / Decoder
- Set dataset path in train.py
```
TRAIN_PATH = DIR_OF_DATASET_IMAGES
```

The training is performed in `train.py`. There are a number of hyperparameters, many corresponding to the augmentation parameters.


### Tensorboard
To visualize the training run the following command and navigate to http://localhost:6006 in your browser.
```bash=
tensorboard --logdir logs
```

## Encoding a Message
The script `encode_image.py` can be used to encode a message into an image or a directory of images. The default model expects a utf-8 encoded secret 200 bit message.

Encode a message into an image:
```bash=
python encode_image.py \
  saved_models/densed2c_pretrained \
  --image test_im.png  \
  --save_dir out/ \
  --secret Changwon_National
```

## Decoding a Message
The script `decode_image.py` can be used to decode a message from a StegaStamp.

Example usage:
```bash=
python decode_image.py \
  saved_models/densed2c_pretrained \
  --image out/test_hidden.png
```
Add the `--save_video FILENAME` flag to save out the results.

The `--visualize_detector` flag can be used to visualize the output of the detector network. The mask corresponds to the segmentation mask, the colored polygons are fit to this segmentation mask using a set of heuristics. The detector outputs can noisy and are sensitive to size of the stegastamp. Further optimization of the detection network is not explored in this paper.
