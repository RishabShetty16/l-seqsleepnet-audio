import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import shutil, sys
from datetime import datetime
import h5py
import hdf5storage

from lseqsleepnet import LSeqSleepNet
from config import Config

from sklearn.metrics import accuracy_score
from datagenerator_wrapper import DataGeneratorWrapper
import time

# Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.app.flags.DEFINE_string("eeg_test_data", "./eval_data_eeg.txt", "file containing the list of test EEG data")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Checkpoint directory")
tf.app.flags.DEFINE_string("norm_dir", "./output/", "Directory containing saved normalization params")
tf.app.flags.DEFINE_float("dropout_rnn", 0.75, "Dropout keep probability")
tf.app.flags.DEFINE_integer("nfilter", 32, "Number of filters")
tf.app.flags.DEFINE_integer("nhidden1", 64, "Hidden units epoch encoder")
tf.app.flags.DEFINE_integer("attention_size", 64, "Attention size")
tf.app.flags.DEFINE_integer("nhidden2", 64, "Hidden units sequence encoder")
tf.app.flags.DEFINE_integer("sub_seq_len", 10, "Subsequence length")
tf.app.flags.DEFINE_integer("nsubseq", 10, "Number of subsequences")
tf.app.flags.DEFINE_integer("dualrnn_blocks", 1, "Number of dual rnn blocks")
tf.app.flags.DEFINE_float("gpu_usage", 0.5, "GPU memory fraction")

FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

out_path = os.path.abspath(os.path.join(os.path.curdir, FLAGS.out_dir))
checkpoint_path = os.path.abspath(os.path.join(out_path, FLAGS.checkpoint_dir))
norm_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.norm_dir))

if not os.path.isdir(out_path): os.makedirs(out_path)
if not os.path.isdir(checkpoint_path): os.makedirs(checkpoint_path)

config = Config()
config.dropout_rnn = FLAGS.dropout_rnn
config.sub_seq_len = FLAGS.sub_seq_len
config.nfilter = FLAGS.nfilter
config.nhidden1 = FLAGS.nhidden1
config.nhidden2 = FLAGS.nhidden2
config.attention_size = FLAGS.attention_size
config.nsubseq = FLAGS.nsubseq
config.dualrnn_blocks = FLAGS.dualrnn_blocks
config.nchannel = 1

# Load saved normalization params from disk (no need to reload train data)
print("Loading normalization params from disk...")
mean_x2 = np.load(os.path.join(norm_dir, 'mean_x2.npy'))
std_x2  = np.load(os.path.join(norm_dir, 'std_x2.npy'))
mean_mel = np.load(os.path.join(norm_dir, 'mean_mel.npy'))
std_mel  = np.load(os.path.join(norm_dir, 'std_mel.npy'))
print("Normalization params loaded!")

# Load ONLY test data
print("Loading test data...")
test_gen_wrapper = DataGeneratorWrapper(
    eeg_filelist=os.path.abspath(FLAGS.eeg_test_data),
    num_fold=1,
    data_shape_2=[config.frame_seq_len, config.ndim],
    seq_len=config.sub_seq_len * config.nsubseq,
    shuffle=False)
test_gen_wrapper.set_eeg_normalization_params(mean_x2, std_x2)
test_gen_wrapper.new_subject_partition()
test_gen_wrapper.next_fold()
test_gen_wrapper.gen.normalize_mel(mean_mel, std_mel)
print("Test data loaded!")

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.gpu_usage,
        allow_growth=True)
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=gpu_options,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        net = LSeqSleepNet(config=config)

        out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.out_dir))
        print("Writing to {}\n".format(out_dir))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        saver = tf.train.Saver(tf.all_variables())
        best_dir = os.path.join(checkpoint_path, "best_model_acc")
        saver.restore(sess, best_dir)
        print("Model loaded!")

        def _reshape_batch(x_batch, mel_batch, y_batch):
            x_shape = x_batch.shape
            y_shape = y_batch.shape
            a_shape = mel_batch.shape
            x = np.zeros(x_shape[:1] + (config.nsubseq, config.sub_seq_len) + x_shape[2:] + (1,))
            y = np.zeros(y_shape[:1] + (config.nsubseq, config.sub_seq_len) + y_shape[2:])
            a = np.zeros(a_shape[:1] + (config.nsubseq, config.sub_seq_len) + a_shape[2:])
            for s in range(config.nsubseq):
                x[:, s, :, :, :, 0] = x_batch[:, s * config.sub_seq_len:(s + 1) * config.sub_seq_len]
                y[:, s] = y_batch[:, s * config.sub_seq_len:(s + 1) * config.sub_seq_len]
                a[:, s] = mel_batch[:, s * config.sub_seq_len:(s + 1) * config.sub_seq_len]
            return x, a, y

        def _make_seq_lens(batch_size):
            fsl = np.ones(batch_size * config.sub_seq_len * config.nsubseq, dtype=int) * config.frame_seq_len
            ssl = np.ones(batch_size * config.nsubseq, dtype=int) * config.sub_seq_len
            isl = np.ones(batch_size * config.sub_seq_len, dtype=int) * config.nsubseq
            return fsl, ssl, isl

        def dev_step(x_batch, mel_batch, y_batch):
            x, a, y = _reshape_batch(x_batch, mel_batch, y_batch)
            fsl, ssl, isl = _make_seq_lens(len(x_batch))
            feed_dict = {
                net.input_x: x,
                net.input_mel: a,
                net.input_y: y,
                net.dropout_rnn: 1.0,
                net.inter_subseq_len: isl,
                net.sub_seq_len: ssl,
                net.frame_seq_len: fsl,
                net.istraining: 0
            }
            output_loss, total_loss, yhat, score = sess.run(
                [net.output_loss, net.loss, net.prediction, net.score], feed_dict)
            return output_loss, total_loss, yhat, score

        def _evaluate(gen):
            output_loss = 0
            total_loss = 0
            yhat = np.zeros([len(gen.data_index), config.sub_seq_len * config.nsubseq])
            score = np.zeros([len(gen.data_index), config.sub_seq_len * config.nsubseq, config.nclass])
            factor = 4  # reduced from 20*4=80 to prevent OOM with dual encoder

            num_batch_per_epoch = np.floor(
                len(gen.data_index) / (factor * config.batch_size)).astype(np.uint32)
            test_step = 1

            while test_step < num_batch_per_epoch:
                x_batch, mel_batch, y_batch, label_batch_ = gen.next_batch(factor * config.batch_size)
                output_loss_, total_loss_, yhat_, score_ = dev_step(x_batch, mel_batch, y_batch)
                output_loss += output_loss_
                total_loss += total_loss_
                for s in range(config.nsubseq):
                    yhat[(test_step - 1) * factor * config.batch_size:
                         test_step * factor * config.batch_size,
                         s * config.sub_seq_len:(s + 1) * config.sub_seq_len] = yhat_[:, s]
                    score[(test_step - 1) * factor * config.batch_size:
                          test_step * factor * config.batch_size,
                          s * config.sub_seq_len:(s + 1) * config.sub_seq_len] = score_[:, s]
                test_step += 1

            if gen.pointer < len(gen.data_index):
                actual_len, x_batch, mel_batch, y_batch, label_batch_ = gen.rest_batch(factor * config.batch_size)
                output_loss_, total_loss_, yhat_, score_ = dev_step(x_batch, mel_batch, y_batch)
                output_loss += output_loss_
                total_loss += total_loss_
                for s in range(config.nsubseq):
                    yhat[(test_step - 1) * factor * config.batch_size:len(gen.data_index),
                         s * config.sub_seq_len:(s + 1) * config.sub_seq_len] = yhat_[:, s]
                    score[(test_step - 1) * factor * config.batch_size:len(gen.data_index),
                          s * config.sub_seq_len:(s + 1) * config.sub_seq_len] = score_[:, s]

            yhat = yhat + 1
            return yhat, score, output_loss, total_loss

        # Run evaluation
        start_time = time.time()
        gen = test_gen_wrapper.gen
        yhat_, score_, output_loss_, total_loss_ = _evaluate(gen)

        N = int(np.sum(test_gen_wrapper.file_sizes) -
                (config.sub_seq_len * config.nsubseq - 1) * len(test_gen_wrapper.file_sizes))
        yhat = np.zeros([N, config.sub_seq_len * config.nsubseq])
        y = np.zeros([N, config.sub_seq_len * config.nsubseq])
        score = np.zeros([N, config.sub_seq_len * config.nsubseq, config.nclass])

        yhat[:len(gen.data_index)] = yhat_
        score[:len(gen.data_index)] = score_

        for n in range(config.sub_seq_len * config.nsubseq):
            y[:len(gen.data_index), n] = gen.label[
                gen.data_index - (config.sub_seq_len * config.nsubseq - 1) + n]

        test_acc = np.zeros([config.sub_seq_len * config.nsubseq])
        for n in range(config.sub_seq_len * config.nsubseq):
            test_acc[n] = accuracy_score(yhat[:, n], y[:, n])

        end_time = time.time()
        print(f"\nTest accuracy: {np.mean(test_acc):.4f}")
        print(f"Test time: {end_time - start_time:.1f}s")

        with open(os.path.join(out_dir, "test_time.txt"), "a") as f:
            f.write("{:g}\n".format(end_time - start_time))

        hdf5storage.savemat(
            os.path.join(out_path, "test_ret.mat"),
            {'yhat': yhat, 'acc': test_acc, 'score': score,
             'output_loss': output_loss_, 'total_loss': total_loss_},
            format='7.3')

        print("Results saved!")
        test_gen_wrapper.gen.reset_pointer()