# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from espnet(https://github.com/espnet/espnet)
"""Length regulator related modules."""
import numpy as np
import paddle
from paddle import nn


class LengthRegulator(nn.Layer):
    """Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or
    phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.

        """
        super().__init__()
        self.pad_value = pad_value

    # expand_numpy is faster than expand
    def expand_numpy(self, encodings: paddle.Tensor,
                     durations: paddle.Tensor) -> paddle.Tensor:
        """
        encodings: (B, T, C)
        durations: (B, T)
        """
        print ("reference expand_numpy")
        batch_size, t_enc = durations.shape
        durations = durations.numpy()
        slens = np.sum(durations,axis=1)
        t_dec = np.max(slens)
        M = np.zeros([batch_size, t_dec, t_enc])
        for i in range(batch_size):
            k = 0
            for j in range(t_enc):
                d = durations[i, j]
                M[i, k:k + d, j] = 1
                k += d
        M = paddle.to_tensor(M, dtype=encodings.dtype)
        encodings = paddle.matmul(M, encodings)
        return encodings,paddle.to_tensor(slens,dtype=paddle.int64)
    
    def expand_batch(self, encodings: paddle.Tensor,
               durations: paddle.Tensor) -> paddle.Tensor:
        """
        encodings: (B, T, C)
        durations: (B, T)
        """
        print ("reference expand")
        batch_size, t_enc = paddle.shape(durations)
        slens = paddle.sum(durations, -1)
        t_dec = paddle.max(slens)
        t_dec_1 = t_dec + 1
        #print ("1*****batch_size:{}****t_enc:{}".format(batch_size,t_enc))
        #print ("2----batch_size:{}----t_dec:{}-------".format(batch_size,t_dec))
        flatten_duration_lists = []
        for i in range(batch_size):
            t = paddle.cumsum(durations[i]) + 1
            flatten_duration_lists.append(t)
        M0 = []
        m_batch = 1 * t_enc
        for i in range(batch_size):
            init = paddle.zeros(t_dec_1)
            M_t = paddle.zeros([t_dec_1, m_batch])
            flatten_duration = flatten_duration_lists[i]
            for j in range(m_batch):
                d = flatten_duration[j]
                m = paddle.concat([paddle.ones(d), paddle.zeros(t_dec_1 - d)], axis=0)
                M_t[:, j] = m - init
                init = m
            M0.append(M_t)
        M = paddle.to_tensor(M0)
        M = M[:,1:t_dec_1,:]
        encodings = paddle.matmul(M, encodings)
        return encodings,paddle.to_tensor(slens,dtype=paddle.int64)

    def expand(self, encodings: paddle.Tensor,
               durations: paddle.Tensor) -> paddle.Tensor:
        """
        encodings: (B, T, C)
        durations: (B, T)
        """

        print ("reference expand")
        dtype = encodings.dtype
        batch_size, t_enc = paddle.shape(durations)
        #print ("1*****batch_size:{}****t_enc:{}".format(batch_size,t_enc))
        slens = paddle.sum(durations, -1)
        ## add by hsl
        max_d = paddle.max(slens)
        max_len = max_d
        flatten_duration = paddle.cumsum(paddle.reshape(durations, [batch_size * t_enc])) + 1
        f0 = paddle.zeros(batch_size,dtype=flatten_duration.dtype)
        flatten_duration = paddle.concat([f0,flatten_duration])
        reps_cumsum = flatten_duration.expand([1,1,-1])
        print ("reps_cumsum:{}".format(reps_cumsum.shape))

        range_ = paddle.arange(2000)[None, :, None]
        a = paddle.cast((reps_cumsum[:, :, :-1] <= range_),dtype)
        b = paddle.cast((reps_cumsum[:, :, 1:] > range_),dtype)

        #mult = paddle.add(a,b) -1
        #mult = paddle.abs(mult)
        mult = paddle.multiply(a,b)
        print (mult)
        print ('mult:{}'.format(mult.shape))
        print ('encodings:{}'.format(encodings.shape))
        enc_rep = paddle.matmul(mult, encodings)
        print ('enc_rep:{}'.format(enc_rep.shape))


        return enc_rep,max_d

    def forward(self, xs, ds, alpha=1.0, is_inference=False):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (Tensor(int64)): Batch of durations of each frame (B, T).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).
        """

        if alpha != 1.0:
            assert alpha > 0
            ds = paddle.round(ds.cast(dtype=paddle.float32) * alpha)
        ds = ds.cast(dtype=paddle.int64)
        '''
        from distutils.version import LooseVersion
        from paddlespeech.t2s.modules.nets_utils import pad_list
        # 这里在 paddle 2.2.2 的动转静是不通的
        # if LooseVersion(paddle.__version__) >= "2.3.0" or hasattr(paddle, 'repeat_interleave'):
        # if LooseVersion(paddle.__version__) >= "2.3.0":
        if hasattr(paddle, 'repeat_interleave'):
            repeat = [paddle.repeat_interleave(x, d, axis=0) for x, d in zip(xs, ds)]
            return pad_list(repeat, self.pad_value)
        '''
        if is_inference:
            return self.expand(xs, ds)
        else:
            return self.expand_numpy(xs, ds)
