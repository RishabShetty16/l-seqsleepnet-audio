class Config(object):
    def __init__(self):
        self.sub_seq_len = 10  # subsequence length
        self.nchannel = 1  # number of channels
        self.nclass = 5

        self.nsubseq = 10

        self.learning_rate = 1e-4
        self.l2_reg_lambda = 0.0001
        self.training_epoch = 10*self.sub_seq_len*self.nsubseq
        self.batch_size = 4
        self.evaluate_every = 500
        self.checkpoint_every = 100

        # Original EEG spectrogram size
        self.ndim = 129          # freq bins
        self.frame_seq_len = 29  # time frames per epoch

        self.nhidden1 = 64
        self.lstm_nlayer1 = 1
        self.attention_size = 64
        self.nhidden2 = 64
        self.lstm_nlayer2 = 1

        self.nfilter = 32
        self.nfft = 256
        self.samplerate = 100
        self.lowfreq = 0
        self.highfreq = 50

        self.dropout_rnn = 0.75
        self.fc_size = 512

        self.dualrnn_blocks = 1
        self.early_stop_count = 50

        self.max_eval_steps = 110

        # --- Mel spectrogram branch (sonification) ---
        self.n_mels = 64                    # mel frequency bins
        self.mel_frame_seq_len = 29         # same time frames as original
        self.mel_nfilter = 32               # filterbank filters for mel branch
        # mel encoder output = nhidden1*2 = 128
        # original encoder output = nhidden1*2 = 128
        # fused = 256, projected back to 128 before dual encoder