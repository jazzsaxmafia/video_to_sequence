#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import ipdb

import cv2

from tensorflow.models.rnn import rnn_cell
from keras.preprocessing import sequence

class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.lstm1 = rnn_cell.BasicLSTMCell(dim_hidden)
        self.lstm2 = rnn_cell.BasicLSTMCell(dim_hidden)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        probs = []

        loss = 0.0

        for i in range(self.n_lstm_steps): ## Phase 1 => only read frames
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1( image_emb[:,i,:], state1 )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2( tf.concat(1,[padding, output1]), state2 )

        # Each video might have different length. Need to mask those.
        # But how? Padding with 0 would be enough?
        # Therefore... TODO: for those short videos, keep the last LSTM hidden and output til the end.

        for i in range(self.n_lstm_steps): ## Phase 2 => only generate captions
            if i == 0:
                current_embed = tf.zeros([self.batch_size, self.dim_hidden])
            else:
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i-1])

            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1( padding, state1 )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2( tf.concat(1,[current_embed, output1]), state2 )

            labels = tf.expand_dims(caption[:,i], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat(1, [indices, labels])
            onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:,i]

            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy)
            loss += current_loss

        loss = loss / tf.reduce_sum(caption_mask)
        return loss, video, video_mask, caption, caption_mask, probs


    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_lstm_steps, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_lstm_steps])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(self.n_lstm_steps):
            if i > 0: tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1( image_emb[:,i,:], state1 )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2( tf.concat(1,[padding,output1]), state2 )

        for i in range(self.n_lstm_steps):

            tf.get_variable_scope().reuse_variables()

            if i == 0:
                current_embed = tf.zeros([1, self.dim_hidden])

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1( padding, state1 )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2( tf.concat(1,[current_embed,output1]), state2 )

            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds


############### Global Parameters ###############
video_path = '/media/storage3/Study/data/youtube_videos'
video_data_path='./data/video_corpus.csv'
video_feat_path = '/media/storage3/Study/data/youtube_feats'

vgg16_path = '/home/taeksoo/Package/tensorflow_vgg16/vgg16.tfmodel'

model_path = './models/'
############## Train Parameters #################
dim_image = 4096
dim_hidden= 256
n_frame_step = 80
n_epochs = 1000
batch_size = 100
learning_rate = 0.001
##################################################

def get_video_data(video_data_path, video_feat_path, train_ratio=0.9):
    video_data = pd.read_csv(video_data_path, sep=',')
    video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(row['Start'])+'_'+str(row['End'])+'.avi.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unique_filenames = video_data['video_path'].unique()
    train_len = int(len(unique_filenames)*train_ratio)

    train_vids = unique_filenames[:train_len]
    test_vids = unique_filenames[train_len:]

    train_data = video_data[video_data['video_path'].map(lambda x: x in train_vids)]
    test_data = video_data[video_data['video_path'].map(lambda x: x in test_vids)]

    return train_data, test_data

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

def train():
    train_data, _ = get_video_data(video_data_path, video_feat_path, train_ratio=0.9)
    captions = train_data['Description'].values
    captions = map(lambda x: x.replace('.', ''), captions)
    captions = map(lambda x: x.replace(',', ''), captions)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=10)

    np.save('./data/ixtoword', ixtoword)

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            bias_init_vector=bias_init_vector)

    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_model()
    sess = tf.InteractiveSession()

    saver = tf.train.Saver(max_to_keep=10)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    tf.initialize_all_variables().run()

    for epoch in range(n_epochs):
        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.ix[index]

        current_train_data = train_data.groupby('video_path').apply(lambda x: x.irow(np.random.choice(len(x))))
        current_train_data = current_train_data.reset_index(drop=True)

        for start,end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data), batch_size)):

            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values

            current_feats = np.zeros((batch_size, n_frame_step, dim_image))
            current_feats_vals = map(lambda vid: np.load(vid), current_videos)

            current_video_masks = np.zeros((batch_size, n_frame_step))

            for ind,feat in enumerate(current_feats_vals):
                current_feats[ind][:len(current_feats_vals[ind])] = feat
                current_video_masks[ind][:len(current_feats_vals[ind])] = 1

            current_captions = current_batch['Description'].values
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_frame_step-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            probs_val = sess.run(tf_probs, feed_dict={
                tf_video:current_feats,
                tf_caption: current_caption_matrix
                })

            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_video: current_feats,
                        tf_video_mask : current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })

            print loss_val
        if np.mod(epoch, 100) == 0:
            print "Epoch ", epoch, " is done. Saving the model ..."
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

def test(model_path='models/model-900', video_feat_path=video_feat_path):

    train_data, test_data = get_video_data(video_data_path, video_feat_path, train_ratio=0.9)
    test_videos = test_data['video_path'].unique()
    ixtoword = pd.Series(np.load('./data/ixtoword.npy').tolist())

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            bias_init_vector=None)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    for video_feat_path in test_videos:
        print video_feat_path
        video_feat = np.load(video_feat_path)[None,...]
        video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        probs_val = sess.run(probs_tf, feed_dict={video_tf:video_feat})
        embed_val = sess.run(last_embed_tf, feed_dict={video_tf:video_feat})
        generated_words = ixtoword[generated_word_index]

        punctuation = np.argmax(np.array(generated_words) == '.')+1
        generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        print generated_sentence
        ipdb.set_trace()

    ipdb.set_trace()
