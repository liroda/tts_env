#!/bin/bash
import os,sys,time,re,json
import numpy as np
import onnxruntime
import soundfile as sf
import gradio as gr
import soxbindings as sox


# add fornt path
front_path = r'tts_front_poly/1'
sys.path.append(front_path)
from vits_textfront import VITS_TextFront

class TTS_engine():
    def __init__(self,prody_file,encoder_file,decoder_file):
        front_dict="tts_front_poly/dict"
        phoneid_map_path= os.path.join(front_dict,"mix_phone_id_map.txt")
        jieba_userdict_path = os.path.join(front_dict,"jieba_userdict.txt")
        poly_dict_path = os.path.join(front_dict,"polypinyin.txt")
        bert_dict_path = os.path.join(front_dict,"vocab.txt")
        symbol_pinyin_path = os.path.join(front_dict,"pinyin_symbol.txt")
        pinyin2phone_path = os.path.join(front_dict,"pinyin_phones.txt")
        cmudict_path = os.path.join(front_dict,"cmu_dict.txt")
        polyword_dict_path = os.path.join(front_dict,"polyword.list")
        polymodel_dir=os.path.join(front_path,"poly_g2pw")
        
        self.frontend = VITS_TextFront(
                                  phoneid_map_path =phoneid_map_path,
                                  jieba_userdict_path=jieba_userdict_path,
                                  bert_dict_path = bert_dict_path,
                                  poly_dict_path=poly_dict_path,
                                  symbol_pinyin_path = symbol_pinyin_path ,
                                  pinyin2phone_path = pinyin2phone_path,
                                  cmudict_path = cmudict_path,
                                  polyword_dict_path = polyword_dict_path,
                                  polymodel_dir = polymodel_dir)
        

        #self.prosody = onnxruntime.InferenceSession(prosody_file,providers=[('CUDAExecutionProvider',{'device_id':0})])
        self.prosody = onnxruntime.InferenceSession(prosody_file,providers=[('CPUExecutionProvider')])

        #self.encoder = onnxruntime.InferenceSession(encoder_file,providers=[('CUDAExecutionProvider',{'device_id':0})])
        self.encoder = onnxruntime.InferenceSession(encoder_file,providers=[('CPUExecutionProvider')])

        #self.decoder = onnxruntime.InferenceSession(decoder_file,providers=[('CPUExecutionProvider',{'device_id':0})]) 
        self.decoder = onnxruntime.InferenceSession(decoder_file,providers=[('CPUExecutionProvider')]) 

    def run(self,textstr,spkid):
     
        front_out = self.frontend.get_front_inputids_embeds(textstr)
        x = np.array(front_out[0]['x']).astype(np.int64)
        x_length = np.array(front_out[0]['x_length']).astype(np.int64)
        word_input_ids = np.array(front_out[0]['word_input_ids']).astype(np.int64)
        expand_length = np.array(front_out[0]['expand_length']).astype(np.int64)

        bert_vec = self.prosody.run(None,{'word_input_ids':word_input_ids,'expand_length':expand_length})

        noise_scale = np.array([1],dtype = np.float32)
        length_scale = np.array([1],dtype = np.float32)
        sid = np.array([spkid],dtype = np.int64)
        ort_input_encoder={
                            'x': x,
                            'x_length': x_length,
                            'noise_scale': noise_scale,
                            'length_scale':length_scale,
                            'sid':sid,
                            'bert_vec': bert_vec[0]
                            }
        z_p,y_mask = self.encoder.run(None,ort_input_encoder)

        ort_input_decoder={
                            'sid':sid,
                            'z_p': z_p,
                            'y_mask': y_mask
                            }
        y = self.decoder.run(None,ort_input_decoder)
        return y
        
def tts_callback(text,speed,volume,sid):
    wav = tts_engine.run(text,sid)
    audio_all = wav[0][0]
    audio_all = audio_all * volume
    if speed != 1 :
         tfm = sox.Transformer()
         tfm.set_globals(multithread=False)
         tfm.tempo(speed)
         wav_speed = tfm.build_array(input_array=audio_all,sample_rate_in=8000).squeeze(-1).astype(np.float32).copy()
    else:
         wav_speed = audio_all
    return "Success", (8000, wav_speed)
if __name__ == "__main__":

    with open('config.json','r',encoding="utf-8") as fp:
        config_data = fp.read()
    config_dict = json.loads(config_data)
    #prosody_file = r'prosody/1/prosody.onnx'
    #encoder_file = r'encoder/1/bert-vits-encoder.onnx'
    #decoder_file = r'decoder/1/vits-decoder-batch.onnx'
    prosody_file = config_dict["prosody"]
    encoder_file = config_dict["encoder"]
    decoder_file = config_dict["decoder"]
    tts_engine = TTS_engine(prosody_file,encoder_file,decoder_file)

    app = gr.Blocks()
    with app:
        gr.Markdown("# AIOT 多说话人语音生成 \n\n")
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Row():
                    with gr.Column():
                        textbox = gr.TextArea(label="Text",placeholder="Type your sentences here",value="好的，稍后会给您发送一条信息，您可以点击链接进入运单详情页面自助操作，或关注顺丰速运微信公众号联系在线客服处理，感谢您的配合，再见。",elem_id = f"tts-input")
                        speed_slider = gr.Slider(minimum=0.7, maximum=2, value=1, step=0.1,label='语速调节')
                        volume_slider = gr.Slider(minimum=0.7, maximum=2, value=1, step=0.1,label='音量调节')
                        #speaker_slider = gr.Slider(minimum=0, maximum=27, value=1, step=1,label='支持27种不同音色(男生7 女生19) 男声编号[3,6,8,9,15,19,24,26] 其余女生')
                        spklists = [  i for i in range(0,27)]
                        speaker_dropdown = gr.Dropdown(choices=spklists, value=1, label='支持27种不同音色,不同编号代表不同音色', \
                                info=' 男声7种 女声19种   男声编号[3,6,8,9,15,19,24,26] 其余女声')
                    with gr.Column():
                        text_output = gr.Textbox(label="Message")
                        audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                        btn = gr.Button("Generate!")
                        btn.click(tts_callback,inputs=[textbox,speed_slider,volume_slider,speaker_dropdown],outputs=[text_output,audio_output])
        app.queue().launch(server_name=config_dict["server_name"],server_port=config_dict["server_port"],share='true')
