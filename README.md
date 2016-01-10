# video_to_sequence
* TensorFlow Implementation of [Sequence to Sequence – Video to Text](http://arxiv.org/abs/1505.00487)

### Usage
* First you need to download "Microsoft Video Description Corpus"
 * Set "video_data_path" in download_videos.py accordingly.
 * Download Youtube videos by running "download_videos.py" 
* Secondly, you need to preprocess downloaded videos
 * Set paths in cnn_utils.py and preprocess.py 
 * Sample & extract features by running "preprocessing.py"
* Train: train() in model.py
 * You might need to change the paths in "Global Parameters" area according to your environment
* Test: test() in model.py

![alt tag](https://github.com/jazzsaxmafia/video_to_sequence/blob/master/plane.jpg)

### License
* BSD License
