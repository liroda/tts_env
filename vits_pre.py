import os
import torch
import numpy as np
import argparse
import utils

from bert import TTSProsody
from bert.prosody_tool import pinyin_dict
import librosa
from mel_processing import spectrogram_torch
import traceback
from utils_textfront import sentence_tokenizer,deal_untoken_oov,pinyins_split,get_align


os.makedirs("./data/waves", exist_ok=True)
os.makedirs("./data/berts", exist_ok=True)
os.makedirs("./data/temps", exist_ok=True)


def log(info: str):
    with open(f'./data/prepare.log', "a", encoding='utf-8') as flog:
        print(info, file=flog)


def get_spec(hps, filename):
    audio, sampling_rate = librosa.load(filename,hps.data.sampling_rate)
    if sampling_rate != hps.data.sampling_rate:
        raise ValueError(
            "{} {} SR doesn't match target {} SR".format(
                sampling_rate, hps.data.sampling_rate
            )
        )
    # change by hsl
    max_value = np.abs(audio).max()
    if max_value > 1.0:
        audio_norm =torch.FloatTensor( audio / hps.data.max_wav_value)
    else:
        audio_norm = torch.FloatTensor(audio)

    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    spec = torch.squeeze(spec, 0)
    return spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/bert_vits_8k.json",
        help="JSON file for configuration",
    )
    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config)

    device = torch.device("cuda:0")
    #device = torch.device("cpu")
    prosody = TTSProsody("./bert", device)
    bert_dict = os.path.join("./bert",'vocab.txt')
    worddict = { line.strip():1   for line in open(bert_dict,'r',encoding="utf-8") }

    fo = open(f"./data/all_db06_mix.txt", "r+", encoding='utf-8')
    f_ex = open(f"./data/excpet_lines.txt","w",encoding='utf-8')
    scrips = []
    while (True):
        try:
            message = fo.readline().strip()
            pinyinstr = fo.readline().strip()
        except Exception as e:
            print('nothing of except:', e)
            break
        if (message == None):
            break
        if (message == ""):
            break
        infosub = message.split("\t")
        fileidx = infosub[0]
        message = infosub[1]
        message = message.replace("#1", " ")
        message = message.replace("#2", " ")
        message = message.replace("#3", " ")
        message = message.replace("#4", " ")
        log(f"{fileidx}\t{message}")
        log(f"\t{pinyinstr}")
        
        text_tokenizer = sentence_tokenizer(message)
        text_oov_tokenizer = deal_untoken_oov(text_tokenizer,worddict)
        pinyins = pinyins_split(pinyinstr)
        try:
            #print ("*******************************************************")
            count_phone,phones_item = get_align(text_tokenizer,pinyins,pinyin_dict)
            phone_items_str = ' '.join(phones_item)
        except Exception as e:
            print(f"{fileidx}\t{message}")
            print('except:', e)
            f_ex.write('{}\n'.format(message))
            f_ex.write('{}\n'.format(pinyins))
            continue
         
        try:
            char_embeds = prosody.get_char_embeds(text_oov_tokenizer)
            char_embeds = prosody.expand_for_phone(char_embeds, count_phone)
        except Exception as e:
            print (text_oov_tokenizer)
            print (count_phone)
            error_info = traceback.format_exc()
            print ("error message is {}".format(error_info))
        else:
            char_embeds_path = f"./data/berts/{fileidx}.npy"
            np.save(char_embeds_path, char_embeds, allow_pickle=False)

            wave_path = f"./data/waves/{fileidx}.wav"
            spec_path = f"./data/temps/{fileidx}.spec.pt"
            spec = get_spec(hps, wave_path)

            torch.save(spec, spec_path)
            scrips.append(
                f"./data/waves/{fileidx}.wav|./data/temps/{fileidx}.spec.pt|./data/berts/{fileidx}.npy|{phone_items_str}")

    fo.close()
    f_ex.close()

    fout = open(f'./filelists/all.txt', 'w', encoding='utf-8')
    for item in scrips:
        print(item, file=fout)
    fout.close()
    fout = open(f'./filelists/valid.txt', 'w', encoding='utf-8')
    for item in scrips[-100:]:
        print(item, file=fout)
    fout.close()
    fout = open(f'./filelists/train.txt', 'w', encoding='utf-8')
    for item in scrips[0:-100]:
        print(item, file=fout)
    fout.close()
