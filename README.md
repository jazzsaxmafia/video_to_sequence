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
* Test: test() in model.py
