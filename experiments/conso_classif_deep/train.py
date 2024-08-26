# -*- coding: utf-8 -*-
#
# This script can be used to train any deep learning model on the BigEarthNet. 
#
# To run the code, you need to provide a json file for configurations of the training.
# 
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 23 Dec 2019
# Version: 1.0.1
# Usage: train.py [CONFIG_FILE_PATH]

from __future__ import print_function

SEED = 42

import random as rn
rn.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.set_random_seed(SEED)

print(tf.test.is_gpu_available())

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


from utils import get_metrics

import sys
sys.path.insert(0, '/home/verlyndem/Documents/bigearthnet-models-tf') 

print(sys.path)
import os
import argparse
from BigEarthNet import BigEarthNet
import json
import importlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def run_model(args, storage_path):
    with tf.Session(config=config) as sess:
        iterator = BigEarthNet(
            args['tr_tf_record_files'], 
            args['batch_size'], 
            args['nb_epoch'], 
            args['shuffle_buffer_size'],
            args['label_type']
        ).batch_iterator
        
        nb_iteration = int(np.ceil(float(args['training_size'] * args['nb_epoch']) / args['batch_size']))
        iterator_ins = iterator.get_next()
        

        model = importlib.import_module('models.' + args['model_name']).DNN_model(args['label_type'], args['modality'])
        model.create_network()
        loss = model.define_loss()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=args['learning_rate']).minimize(loss)


        variables_to_save = tf.global_variables()
        _, metric_means, metric_update_ops = get_metrics(model.multi_hot_label, model.predictions, model.probabilities)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        model_saver = tf.train.Saver(max_to_keep=0, var_list=variables_to_save)
        iteration_idx = 0

        if args['fine_tune']:
            model_saver.restore(sess, args['model_file'])
            if 'iteration' in args['model_file']:
                iteration_idx = int(args['model_file'].split('iteration-')[-1])

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(storage_path, 'logs', 'training'), sess.graph)
        val_summary_writer = tf.summary.FileWriter(os.path.join(storage_path, 'logs', 'validation'))
        
        progress_bar = tf.contrib.keras.utils.Progbar(target = nb_iteration) 
        while True:
            try:
                batch_dict = sess.run(iterator_ins)
            except tf.errors.OutOfRangeError:
                break
            _, _, batch_loss, batch_summary = sess.run([train_op, metric_update_ops, loss, summary_op], 
                                                        feed_dict = model.feed_dict(batch_dict, is_training=True))
            iteration_idx += 1
            summary_writer.add_summary(batch_summary, iteration_idx)
            if (iteration_idx % args['save_checkpoint_per_iteration'] == 0) and (iteration_idx >= args['save_checkpoint_after_iteration']):
                model_saver.save(sess, os.path.join(storage_path, 'models', 'iteration'), iteration_idx)
            
            # we do periodic validation after each epoch
            if iteration_idx % int(np.floor(nb_iteration/4)) == 0:
                val_iterator = BigEarthNet(
                    args['val_tf_record_files'], 
                    args['batch_size'], 
                    1, 
                    0,
                    args['label_type']
                    ).batch_iterator
                val_iterator_ins = val_iterator.get_next()
                count_validation = 0
                progress_bar_val = tf.contrib.keras.utils.Progbar(target=int(np.ceil(float(args['val_size']) / args['batch_size'])))
                loss_val = []
                while True:
                    try:
                        batch_dict_val = sess.run(val_iterator_ins)
                        batch_loss_val = sess.run(loss, feed_dict = model.feed_dict(batch_dict_val, is_training=False))
                        loss_val.append(batch_loss_val)
                        count_validation += 1
                        progress_bar_val.update(count_validation)
                    except tf.errors.OutOfRangeError:
                        break
                print('Validation loss: {}'.format(np.mean(loss_val)))
                summary = tf.Summary()
                summary.value.add(tag='loss', simple_value=np.mean(loss_val))
                val_summary_writer.add_summary(summary, iteration_idx)
            progress_bar.update(iteration_idx, values=[('loss', batch_loss)])
        model_saver.save(sess, os.path.join(storage_path, 'models', 'iteration'), iteration_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= 'Training script')
    parser.add_argument('--configs', help= 'json config file')
    parser.add_argument("--storage_path", type=str, required=True)
    parser_args = parser.parse_args()

    print("Entered train.py")

    with open('configs_conso/base.json', 'rb') as f:
        args = json.load(f)

    with open(os.path.realpath(parser_args.configs), 'rb') as f:
        model_args = json.load(f)

    args.update(model_args)
    print(args['modality'])

    run_model(args, parser_args.storage_path)
    
    print("Done")
    print("Results saved in {}".format(parser_args.storage_path))
    print("End of the script")