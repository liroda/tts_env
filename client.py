#!/usr/bin/env python3

import os,sys,requests
import numpy as np
import wave
from tqdm import tqdm 
import time

def generate_wav(outwav,outdir,i):

    wavpath = os.path.join(outdir,'auio_{}.wav'.format(i+1))
    wavfile=wave.open(wavpath,"wb")
    wavfile.setnchannels(1)
    wavfile.setsampwidth(16 // 8)
    wavfile.setframerate(8000)
    wavfile.writeframes(outwav.tobytes())
    wavfile.close()

def generate_in(text):
    request_data = {
    "inputs":[
        {"name":"textstring","datatype":"BYTES","shape":[1],"data":[text]},
        {"name":"noise_scale","datatype":"FP32","shape":[1],"data":[0.2]},
        {"name":"length_scale","datatype":"FP32","shape":[1],"data":[1.0]},
        {"name":"speed","datatype":"FP32","shape":[1],"data":[1.0]},
        {"name":"volume","datatype":"FP32","shape":[1],"data":[1.5]},
        ]
    }
    return request_data

if __name__ == "__main__":

    inputfile,outdir = sys.argv[1:]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    alllines = [ line.strip() for line in open(inputfile,'r',encoding="utf-8") ]

    len1 = len(alllines)

    timefile = '{}/time.log'.format(outdir)
    with open(timefile,'w',encoding="utf-8") as fw:
        for i in tqdm(range(len1)):
            line = alllines[i]
            text = line
            start_time = time.time()
            request_data = generate_in(text)
            print (request_data)
            res = requests.post(url="http://10.202.90.191:8950/v2/models/ensemble_tts_model/infer",json=request_data).json()
            outputs = res["outputs"][0]
            outwav = np.array(res["outputs"][0]["data"],dtype=np.int16)
            end_time = time.time()
            dur_time =  (end_time - start_time)*1000
            generate_wav(outwav,outdir,i)
            fw.write('{} {:.2f}\n'.format(text,dur_time))
