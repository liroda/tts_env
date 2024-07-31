!/usr/bin/env python3

import onnxruntime
import soundfile
import torch
import os,argparse,datetime
import numpy as np
import time

from front.vits_textfront import VITS_TextFront


class OnnxModel_Encoder:
    def __init__(
        self,
        model: str,
    ):
        session_opts = onnxruntime.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 8

        self.session_opts = session_opts

        self.model = onnxruntime.InferenceSession(
            model,
            sess_options=self.session_opts,
            providers=[('CPUExecutionProvider')]
        )
        #providers=[('CUDAExecutionProvider',{'device_id':2})]

    def __call__(self, x,x_length,noise_scale,length_scale):
        """
        Args:
          x:
            A int64 tensor of shape (L,)
        """

        z_p, y_mask = self.model.run(
            [
                self.model.get_outputs()[0].name,
                self.model.get_outputs()[1].name,
            ],
            {
                self.model.get_inputs()[0].name: x,
                self.model.get_inputs()[1].name: x_length,
                self.model.get_inputs()[2].name: noise_scale,
                self.model.get_inputs()[3].name: length_scale,
            },
        )
        return z_p, y_mask


class OnnxModel_Decoder:
    def __init__(
        self,
        model: str,
    ):
        session_opts = onnxruntime.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 8

        self.session_opts = session_opts

        self.model = onnxruntime.InferenceSession(
            model,
            sess_options=self.session_opts,
            providers=[('CPUExecutionProvider')]
        )


    def __call__(self, z_p, y_mask):
        y = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: z_p,
                self.model.get_inputs()[1].name: y_mask,
            },
        )[0]
        return y


def decoder_stream(decoder,z_p,y_mask):
    
    len_z = z_p.shape[2]
    hop_frame = 12
    hop_length=100

    # can not change these parameters
    hop_sample = hop_frame * hop_length
    stream_chunk = 50
    stream_index = 0
    stream_out_wav = []
    #print('frame size:{} hop_length:{} stream_chunk:{}'.format(len_z,hop_length,stream_chunk))

    while (stream_index < len_z):
        if (stream_index == 0):  # start frame
            cut_s = stream_index
            cut_s_wav = 0
        else:
            cut_s = stream_index - hop_frame
            cut_s_wav = hop_sample

        if (stream_index + stream_chunk > len_z - hop_frame):  # end frame
            cut_e = stream_index + stream_chunk
            cut_e_wav = -1
        else:
            cut_e = stream_index + stream_chunk + hop_frame
            cut_e_wav = -1 * hop_sample

        start_time = time.time()
        z_chunk = z_p[:, :, cut_s:cut_e]
        m_chunk = y_mask[:, :, cut_s:cut_e]
        #print ('cut_s:{} cut_e:{}'.format(cut_s,cut_e))
        o_chunk = decoder(z_chunk, m_chunk)
        o_chunk = o_chunk[0,cut_s_wav:cut_e_wav]
        #print ('cut_s_wav:{},cut_e_wav:{} o_chunk:{}'.format(cut_s_wav,cut_e_wav,o_chunk.shape))
        print ('wav stream {}ms'.format(o_chunk.shape[0]/8))
        stream_out_wav.extend(o_chunk)
        print('decoder stream_index:{}'.format(stream_index,datetime.datetime.now()))
        stream_index = stream_index + stream_chunk
        end_time = time.time()
        dur_time = '{}ms'.format((end_time - start_time)*1000)
        print ('dur time:{}'.format(dur_time))
    return stream_out_wav


def decoder_stream_batch(decoder,z_p,y_mask):
    
    len_z = z_p.shape[2]
    hop_frame = 12
    hop_length=100

    # can not change these parameters
    hop_sample = hop_frame * hop_length
    stream_chunk = 50
    stream_index = 0

    flag = 0
    z_chunk_lists=[]
    m_chunk_lists=[]
    
    chunk_start_wav = hop_sample
    chunk_end_wav = -hop_sample

    while (stream_index < len_z):
        if (stream_index == 0):  # start frame
            cut_s = stream_index 
            zp_first_padding = np.zeros([1,192,hop_frame],dtype=np.float32)
            y_mask_first_padding = np.zeros([1,1,hop_frame],dtype=np.float32)
        else:
            cut_s = stream_index - hop_frame
            flag = 1
        

        if (stream_index + stream_chunk > len_z - hop_frame):  # end frame
            cut_e = stream_index + stream_chunk 
            end_pad_length = cut_e + hop_frame -len_z
            cut_e_wav = -1 * end_pad_length * hop_length
            zp_end_padding = np.zeros([1,192,end_pad_length],dtype=np.float32)
            y_mask_end_padding = np.zeros([1,1,end_pad_length],dtype=np.float32)
            flag = 2
        else:
            cut_e = stream_index + stream_chunk + hop_frame
            cut_e_wav = -1 * hop_sample
             

        z_chunk = z_p[:, :, cut_s:cut_e]
        m_chunk = y_mask[:, :, cut_s:cut_e]
        if flag == 0:
            # 首帧的话前端加hop_frame帧
            z_chunk =np.concatenate((zp_first_padding,z_chunk),axis=2)
            m_chunk = np.concatenate((y_mask_first_padding,m_chunk),axis=2)

        elif flag == 2:
            # 尾帧的话后面增加 end_pad_length 长度的帧
            z_chunk =np.concatenate((z_chunk,zp_end_padding),axis=2)
            m_chunk = np.concatenate((m_chunk,y_mask_end_padding),axis=2)

        stream_index = stream_index + stream_chunk

        z_chunk_lists.append(z_chunk)
        m_chunk_lists.append(m_chunk)

    batch_z_chunk = np.concatenate(z_chunk_lists)
    batch_m_chunk = np.concatenate(m_chunk_lists)
    batch_o_chunk = decoder(batch_z_chunk, batch_m_chunk)
    print ("batch_o_chunk:{}".format(batch_o_chunk.shape))


    real_chunk_wav = batch_o_chunk[0:-1,chunk_start_wav:chunk_end_wav].flatten()
    final_chunk_wav  = batch_o_chunk[-1,chunk_start_wav:cut_e_wav].flatten()
    return np.concatenate((real_chunk_wav,final_chunk_wav),axis=0)

def main():
    parser = argparse.ArgumentParser(
        description='Inference code for bert vits models')
    parser.add_argument('--encoder', type=str, required=True)
    parser.add_argument('--decoder', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()
    print("Onnx model path:", args.encoder)
    print("Onnx model path:", args.decoder)
    outdir = args.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    encoder = OnnxModel_Encoder(args.encoder)
    decoder = OnnxModel_Decoder(args.decoder)
    vits = VITS_TextFront(r'cpu',combine=True)


    alllines = [ line.strip() for line in open(args.text,'r',encoding="utf-8")]
    print(datetime.datetime.now())
    for i,line in enumerate(alllines):
        start_front = time.time()
        result = vits.get_front_inputids(line)
        end_front = time.time()
        print ('front compute time {}ms'.format( (end_front - start_front)*1000))
        wav_all =[]
        for rr in result:
            part_input_ids = rr["input"]
            input_length = rr["input_length"]
            noise_scale = np.array([1], dtype=np.float32)
            length_scale = np.array([1], dtype=np.float32)
            start_en = time.time()
            z_p, y_mask = encoder(part_input_ids,input_length,noise_scale,length_scale)
            end_en = time.time()
            print('encoder compute time :{}ms'.format( (end_en - start_en)*1000))
            print('decoder start time:{}'.format(datetime.datetime.now()))
            #stream_out_wav = decoder_stream(decoder,z_p,y_mask)
            stream_out_wav = decoder_stream_batch(decoder,z_p,y_mask)
            wav_all.extend(stream_out_wav)
        wav_all = np.asarray(wav_all)
        soundfile.write('{}/onnx_stream_{}_hop12.wav'.format(outdir,i),wav_all,8000)

if __name__ == "__main__":
    main()
