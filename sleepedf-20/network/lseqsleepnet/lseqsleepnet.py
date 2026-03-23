import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from nn_basic_layers import *
from filterbank_shape import FilterbankShape
from ops import *

class LSeqSleepNet(object):

    def __init__(self, config):
        self.config = config

        # --- Original EEG spectrogram input ---
        # (batch, nsubseq, sub_seq_len, frame_seq_len, ndim, nchannel)
        self.input_x = tf.placeholder(tf.float32,
            [None, self.config.nsubseq, self.config.sub_seq_len,
             self.config.frame_seq_len, self.config.ndim, self.config.nchannel],
            name="input_x")

        # --- NEW: Mel spectrogram input (sonification branch) ---
        # (batch, nsubseq, sub_seq_len, mel_time_frames, n_mels)
        self.input_mel = tf.placeholder(tf.float32,
            [None, self.config.nsubseq, self.config.sub_seq_len,
             self.config.mel_frame_seq_len, self.config.n_mels],
            name="input_mel")

        self.input_y = tf.placeholder(tf.float32,
            [None, self.config.nsubseq, self.config.sub_seq_len, self.config.nclass],
            name="input_y")

        self.dropout_rnn = tf.placeholder(tf.float32, name="dropout_rnn")
        self.istraining = tf.placeholder(tf.bool, name='istraining')
        self.frame_seq_len = tf.placeholder(tf.int32, [None])
        self.inter_subseq_len = tf.placeholder(tf.int32, [None])
        self.sub_seq_len = tf.placeholder(tf.int32, [None])

        self.filtershape = FilterbankShape()

        # ------------------------------------------------------------------ #
        # 1. Original EEG Spectrogram Branch
        # ------------------------------------------------------------------ #
        x = tf.reshape(self.input_x,
                       [-1, self.config.ndim, self.config.nchannel])
        processed_x = self.preprocessing(x)
        processed_x = tf.reshape(processed_x,
            [-1, self.config.frame_seq_len,
             self.config.nfilter * self.config.nchannel])
        # epoch_x shape: (-1, nhidden1*2) = (-1, 128)
        epoch_x = self.epoch_encoder(processed_x, scope_name="seq_frame_rnn_layer",
                                     attn_scope="frame_attention_layer")

        # ------------------------------------------------------------------ #
        # 2. Mel Spectrogram Branch (Sonification — separate encoder)
        # ------------------------------------------------------------------ #
        # Reshape mel input: (batch*nsubseq*sub_seq_len, mel_time_frames, n_mels)
        mel_flat = tf.reshape(self.input_mel,
            [-1, self.config.mel_frame_seq_len, self.config.n_mels])

        # Mel filterbank projection layer (learnable, like original filterbank)
        with tf.variable_scope("mel_filterbank_layer", reuse=tf.AUTO_REUSE):
            # Project n_mels -> mel_nfilter
            W_mel = tf.get_variable('W_mel',
                shape=[self.config.n_mels, self.config.mel_nfilter],
                initializer=tf.random_normal_initializer())
            W_mel = tf.sigmoid(W_mel)  # non-negative constraint
            # mel_flat: (-1, mel_time_frames, n_mels)
            mel_proj = tf.reshape(
                tf.matmul(
                    tf.reshape(mel_flat, [-1, self.config.n_mels]),
                    W_mel),
                [-1, self.config.mel_frame_seq_len, self.config.mel_nfilter])

        # mel_x shape: (-1, nhidden1*2) = (-1, 128)
        mel_x = self.epoch_encoder(mel_proj, scope_name="mel_frame_rnn_layer",
                                   attn_scope="mel_attention_layer")

        # ------------------------------------------------------------------ #
        # 3. Late Fusion: concatenate original + mel embeddings
        # ------------------------------------------------------------------ #
        # fused shape: (-1, nhidden1*4) = (-1, 256)
        fused = tf.concat([epoch_x, mel_x], axis=1)

        # Project back to nhidden1*2 = 128 so dual encoder is unchanged
        with tf.variable_scope("fusion_projection"):
            fused = fc(fused,
                       self.config.nhidden1 * 4,
                       self.config.nhidden1 * 2,
                       name="fusion_fc", relu=True)

        # ------------------------------------------------------------------ #
        # 4. Long sequence modelling (unchanged)
        # ------------------------------------------------------------------ #
        fused = tf.reshape(fused,
            [-1, self.config.nsubseq, self.config.sub_seq_len,
             self.config.nhidden1 * 2])
        seq_x = self.dual_sequence_encoder(fused, self.config.dualrnn_blocks)

        # ------------------------------------------------------------------ #
        # 5. Output FC layers (unchanged)
        # ------------------------------------------------------------------ #
        with tf.variable_scope("output_layer"):
            X_out = tf.reshape(seq_x, [-1, self.config.nhidden1 * 2])
            fc1 = fc(X_out, self.config.nhidden1 * 2, self.config.fc_size,
                     name="fc1", relu=True)
            fc1 = dropout(fc1, self.dropout_rnn)
            fc2 = fc(fc1, self.config.fc_size, self.config.fc_size,
                     name="fc2", relu=True)
            fc2 = dropout(fc2, self.dropout_rnn)
            self.score = fc(fc2, self.config.fc_size, self.config.nclass,
                            name="output", relu=False)
            self.prediction = tf.argmax(self.score, 1, name="pred")
            self.score = tf.reshape(self.score,
                [-1, self.config.nsubseq, self.config.sub_seq_len,
                 self.config.nclass])
            self.prediction = tf.reshape(self.prediction,
                [-1, self.config.nsubseq, self.config.sub_seq_len])

        # ------------------------------------------------------------------ #
        # 6. Loss
        # ------------------------------------------------------------------ #
        self.output_loss = 0.0
        with tf.name_scope("output-loss"):
            y = tf.reshape(self.input_y, [-1, self.config.nclass])
            logit = tf.reshape(self.score, [-1, self.config.nclass])
            self.output_loss += tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=y, logits=logit), axis=[0])
            self.output_loss /= (self.config.nsubseq * self.config.sub_seq_len)

        with tf.name_scope("l2_loss"):
            vars = tf.trainable_variables()
            except_vars_eeg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='seq_filterbank-layer-eeg')
            except_vars_eog = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='seq_filterbank-layer-eog')
            except_vars_emg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='seq_filterbank-layer-emg')
            # Also exclude mel filterbank from L2
            except_vars_mel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='mel_filterbank_layer')
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars
                                if v not in except_vars_eeg
                                and v not in except_vars_eog
                                and v not in except_vars_emg
                                and v not in except_vars_mel])
            self.loss = self.output_loss + self.config.l2_reg_lambda * l2_loss

        # ------------------------------------------------------------------ #
        # 7. Accuracy
        # ------------------------------------------------------------------ #
        self.accuracy = []
        with tf.name_scope("accuracy"):
            y = tf.reshape(self.input_y,
                [-1, self.config.nsubseq * self.config.sub_seq_len,
                 self.config.nclass])
            yhat = tf.reshape(self.prediction,
                [-1, self.config.nsubseq * self.config.sub_seq_len])
            for i in range(self.config.nsubseq * self.config.sub_seq_len):
                correct_prediction_i = tf.equal(
                    yhat[:, i],
                    tf.argmax(tf.squeeze(y[:, i, :]), 1))
                accuracy_i = tf.reduce_mean(
                    tf.cast(correct_prediction_i, "float"),
                    name="accuracy-%s" % i)
                self.accuracy.append(accuracy_i)

    def preprocessing(self, input):
        """Original EEG filterbank preprocessing."""
        Wbl = tf.constant(
            self.filtershape.lin_tri_filter_shape(
                nfilt=self.config.nfilter,
                nfft=self.config.nfft,
                samplerate=self.config.samplerate,
                lowfreq=self.config.lowfreq,
                highfreq=self.config.highfreq),
            dtype=tf.float32,
            name="W-filter-shape-eeg")

        with tf.variable_scope("seq_filterbank-layer-eeg", reuse=tf.AUTO_REUSE):
            Xeeg = tf.reshape(tf.squeeze(input[:, :, 0]), [-1, self.config.ndim])
            Weeg = tf.get_variable('Weeg',
                shape=[self.config.ndim, self.config.nfilter],
                initializer=tf.random_normal_initializer())
            Weeg = tf.sigmoid(Weeg)
            Wfb_eeg = tf.multiply(Weeg, Wbl)
            HWeeg = tf.matmul(Xeeg, Wfb_eeg)

        if self.config.nchannel > 1:
            with tf.variable_scope("seq_filterbank-layer-eog", reuse=tf.AUTO_REUSE):
                Xeog = tf.reshape(tf.squeeze(input[:, :, 1]), [-1, self.config.ndim])
                Weog = tf.get_variable('Weog',
                    shape=[self.config.ndim, self.config.nfilter],
                    initializer=tf.random_normal_initializer())
                Weog = tf.sigmoid(Weog)
                Wfb_eog = tf.multiply(Weog, Wbl)
                HWeog = tf.matmul(Xeog, Wfb_eog)

        if self.config.nchannel > 2:
            with tf.variable_scope("seq_filterbank-layer-emg", reuse=tf.AUTO_REUSE):
                Xemg = tf.reshape(tf.squeeze(input[:, :, 2]), [-1, self.config.ndim])
                Wemg = tf.get_variable('Wemg',
                    shape=[self.config.ndim, self.config.nfilter],
                    initializer=tf.random_normal_initializer())
                Wemg = tf.sigmoid(Wemg)
                Wfb_emg = tf.multiply(Wemg, Wbl)
                HWemg = tf.matmul(Xemg, Wfb_emg)

        if self.config.nchannel > 2:
            X2 = tf.concat([HWeeg, HWeog, HWemg], axis=1)
        elif self.config.nchannel > 1:
            X2 = tf.concat([HWeeg, HWeog], axis=1)
        else:
            X2 = HWeeg
        return X2

    def epoch_encoder(self, input, scope_name, attn_scope):
        """
        BiLSTM + attention epoch encoder.
        Used for both original spectrogram and mel spectrogram branches.
        input shape: (-1, time_frames, feat_dim)
        output shape: (-1, nhidden1*2)
        """
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
            fw_cell, bw_cell = bidirectional_recurrent_layer_bn_new(
                self.config.nhidden1,
                self.config.lstm_nlayer1,
                seq_len=self.config.frame_seq_len,
                is_training=self.istraining,
                input_keep_prob=self.dropout_rnn,
                output_keep_prob=self.dropout_rnn)
            rnn_out, _ = bidirectional_recurrent_layer_output_new(
                fw_cell, bw_cell, input, self.frame_seq_len, scope=scope)

        with tf.variable_scope(attn_scope, reuse=tf.AUTO_REUSE):
            attn_out, _ = attention(rnn_out, self.config.attention_size)

        return attn_out

    def residual_rnn(self, input, seq_len,
                     in_dropout=1.0, out_dropout=1.0, name='rnn_res'):
        _, nseq, dim = input.get_shape().as_list()
        with tf.variable_scope(name) as scope:
            fw_cell, bw_cell = bidirectional_recurrent_layer_bn_new(
                self.config.nhidden2,
                self.config.lstm_nlayer2,
                seq_len=nseq,
                is_training=self.istraining,
                input_keep_prob=in_dropout,
                output_keep_prob=out_dropout)
            rnn_out, _ = bidirectional_recurrent_layer_output_new(
                fw_cell, bw_cell, input, seq_len, scope=scope)
            rnn_out = fc(tf.reshape(rnn_out, [-1, self.config.nhidden2 * 2]),
                         self.config.nhidden2 * 2, dim,
                         name="fc", relu=False)
            rnn_out = tf.keras.layers.LayerNormalization()(rnn_out)
            rnn_out = tf.reshape(rnn_out, [-1, nseq, dim]) + input
        return rnn_out

    def dual_sequence_encoder(self, input, N):
        _, nsubseq, subseq_len, dim = input.get_shape().as_list()
        for n in range(N):
            input = tf.reshape(input, [-1, subseq_len, dim])
            rnn1_out = self.residual_rnn(
                input, self.sub_seq_len,
                in_dropout=1.0 if n == 0 else self.dropout_rnn,
                out_dropout=self.dropout_rnn,
                name='intra_chunk_rnn_' + str(n + 1))
            rnn1_out = tf.reshape(rnn1_out, [-1, nsubseq, subseq_len, dim])
            rnn1_out = tf.transpose(rnn1_out, perm=[0, 2, 1, 3])
            rnn1_out = tf.reshape(rnn1_out, [-1, nsubseq, dim])

            rnn2_out = self.residual_rnn(
                rnn1_out, self.inter_subseq_len,
                in_dropout=self.dropout_rnn,
                out_dropout=self.dropout_rnn,
                name='inter_chunk_rnn_' + str(n + 1))
            rnn2_out = tf.reshape(rnn2_out, [-1, subseq_len, nsubseq, dim])
            rnn2_out = tf.transpose(rnn2_out, perm=[0, 2, 1, 3])
            rnn2_out = tf.reshape(rnn2_out, [-1, subseq_len, dim])
            input = rnn2_out

        input = tf.reshape(input, [-1, nsubseq, subseq_len, dim])
        input = dropout(input, self.dropout_rnn)
        return input