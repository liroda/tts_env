mport sys, os
import wave, glob,datetime
import numpy as np
from scipy.io.wavfile import read
from tqdm import tqdm
import logging
from collections import defaultdict

def read_wav(path):
    src_rate, data = read(path)
    # 检查数据类型
    assert data.dtype in ('float32', 'int16'), 'data type error'
    # 数据通道数
    src_channels = 1 if len(data.shape)==1 else data.shape[1]

    if data.dtype == 'float32':
        # 32bit --> 16bit
        # data = data/1.414
        data = (data*32767).astype(np.int16)
    if src_channels > 1:
        data = data.astype(np.int16)
       

    return src_rate, data


def snr(wavc, segs, ignore_jump=True): 

    wav,c = wavc
   
    rate, data = read(wav)
    # add 2021-05-12
    wav_length = data.shape[0] / rate

    dur = len(data) / rate
    # 提取c 对应通道数据 
    if c == "1":
        datac = data[:,0]
        logging.info("all 0 channel {}".format(np.average(np.square(data[:,0]))))
    else:
        datac = data[:,1]
        logging.info("all 1 channel {}".format(np.average(np.square(data[:,1]))))

    if len(segs[2:]) > 1:
        speech_dots = np.concatenate([np.square(datac[int(seg[0]*rate): int(seg[1]*rate)].astype(np.int64)) for seg in segs[2:]])
        speech_pwr = np.average(speech_dots)

        speech_abs = np.concatenate([np.abs(datac[int(seg[0]*rate): int(seg[1]*rate)].astype(np.int64)) for seg in segs[2:]])
        speech_am = np.average(speech_abs)
        logging.info("signal average ampltude:{}".format(int(speech_am)))
    else:
        speech_pwr = 0.00000001
        speech_am = 0.000000001
    
    noise_dots = np.array([])

    for i_seg in np.arange(len(segs)-1):
        noise_dots = np.concatenate([noise_dots, np.square(datac[int(segs[i_seg][1]*rate+1):int(segs[i_seg+1][0]*rate-1)].astype(np.int64))])
    noise_dots = np.concatenate([noise_dots, np.square(datac[int(segs[-1][1]*rate+1):].astype(np.int64))])
    noise_pwr = np.average(noise_dots)
    
    
    return round(10*np.log10(speech_pwr), 2), round(10*np.log10(noise_pwr), 2), round(10*np.log10(speech_pwr/noise_pwr), 2),round(wav_length,2),int(speech_am)

seg_dir,channel= sys.argv[1:]

logging.basicConfig(filename='snr.log',format='%(asctime)s:%(message)s',level=logging.INFO)
wav_segs = {}
speech_durs ={}
with open(os.path.join(seg_dir, 'segments')) as fp:
    for line in fp:
        utt, wav, begin, end = line.strip().split()
        dur = float(end) - float(begin)
        
        if wav not in wav_segs:
            wav_segs[wav] = []
            speech_durs[wav] = 0
        wav_segs[wav].append([float(begin), float(end)])
        speech_durs[wav]+= dur

wav_path = {}
wav2channel = {}
with open(os.path.join(seg_dir, 'wav.scp')) as fp:
    for line in fp:
        lst = line.strip().split()
        path = lst[1] if len(lst)==2 else lst[2]
        if "remix" in line:
            wav2channel[lst[0]] = (path,line.split()[-2]) 
        wav_path[lst[0]] = path
record2info = defaultdict(dict)
with open(os.path.join(seg_dir, 'snr_valid.txt'+'-'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), 'w') as fp:
    fp.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n' .format("wav","wavtime","channel_0_valid","channel_1_valid","channel_0_amplitue","channel_1_amplitude","channel_0_snr","channel_1_snr"))
    for wav, segs in tqdm(wav_segs.items()):
        if channel == "2":
            logging.info("deal wav {}".format(wav))
            try :
                speech_level, noise_level, snr_ratio,wav_length,speech_am = snr(wav2channel[wav], segs)
            except Exception as e:
                 logging.error(type(e),e) 
        
            else:
                valid_ratio = round(speech_durs[wav]*100 / wav_length,2)
                record_wav,chstr = wav.split('-')
                record2info[record_wav]["wavtime"] = wav_length
                if chstr == "A":
                    record2info[record_wav]["0"] = [snr_ratio,valid_ratio,speech_am]
                elif chstr == "B":
                    record2info[record_wav]["1"] = [snr_ratio,valid_ratio,speech_am]
    for key,dictvalue  in record2info.items():
        record = key 
        wavtime = dictvalue["wavtime"]  
        channel0_snr,channel0_valid,channel0_am = dictvalue["0"]
        channel1_snr,channel1_valid,channel1_am = dictvalue["1"]
        
        fp.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\n'.format(record,wavtime,channel0_valid, channel1_valid,channel0_am, channel1_am,channel0_snr,channel1_snr))
