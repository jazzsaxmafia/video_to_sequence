#-*- coding: utf-8 -*-
'''
According to the paper, the authors extracted upto 80 frames from each video,
they did not mention if they grabbed first 80 frames, or sampled 80 frames with same intervals,
but anyway I did the latter.
'''
import cv2
import os
import ipdb
import numpy as np
import pandas as pd
import skimage
from cnn_util import *

def preprocess_frame(frame):
    short_edge = min(frame.shape[:2])
    yy = int((frame.shape[0] - short_edge) / 2)
    xx = int((frame.shape[1] - short_edge) / 2)
    crop_img = frame[yy : yy + short_edge, xx : xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (224, 224))

    return resized_img

def main():
    num_frames = 80
    vgg_model = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
    vgg_deploy = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'
    video_save_path = '/media/storage3/Study/data/youtube_videos'
    videos = os.listdir(video_save_path)

    cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=224, height=224)

    for video in videos:
        video_fullpath = os.path.join(video_save_path, video)
        cap  = cv2.VideoCapture( video_fullpath )

        frame_count = 0
        frame_list = []

        while True:
            ret, frame = cap.read()
            if ret is False:
                break

            frame_list.append(frame)
            frame_count += 1

        frame_list = np.array(frame_list)

        if frame_count > 80:
            frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
            frame_list = frame_list[frame_indices]

        cropped_frame_list = np.array(map(lambda x: preprocess_frame(x), frame_list))
        feats = cnn.get_features(cropped_frame_list)

        save_full_path = os.path.join(video_save_path, video + '.npy')
        np.save(save_full_path, feats)

if __name__=="__main__":
    main()
