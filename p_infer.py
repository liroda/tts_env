#!/usr/bin/env python3

''' 
   导入g2pW 
   bert-base-chinese
   vocab.txt
'''

import os,sys,json,time
from typing import Dict,List,Tuple
import numpy as np
import onnxruntime
import pickle
import torch

DEVICE='cuda'

class G2PW_Process:
    def __init__(self,
                 model_dir=r'/models/tts_front_poly/1/poly_g2pw',
                 style: str='bopomofo',
                 enable_non_tradional_chinese: bool=True):

        self.use_mask = True
        self.window_size = 30 
        bert_dictpath = os.path.join(model_dir,'dict/vocab.txt')
        alllines_polybert = [ line.strip() for line in open(bert_dictpath,'r',encoding="utf-8") ]
        self.bert_dict = { line:i  for i,line in enumerate(alllines_polybert) }
        self.enable_non_tradional_chinese = enable_non_tradional_chinese

        # 简体转繁体字典
        if self.enable_non_tradional_chinese:
            self.s2t_dict = {}
            for line in open(os.path.join(model_dir,'dict/bert-base-chinese_s2t_dict.txt'), 'r',encoding="utf-8").read().strip().split('\n'):
                s_char, t_char = line.split('\t')
                self.s2t_dict[s_char] = t_char

        # dict
        polyphonic_chars_path = os.path.join(model_dir,'dict/POLYPHONIC_CHARS.txt')
        self.polyphonic_chars = [line.split('\t') for line in open(polyphonic_chars_path,'r', encoding='utf-8').read().strip().split('\n')]
        self.labels,self.char2phonemes = self.get_phoneme_labels(polyphonic_chars=self.polyphonic_chars)
        self.chars = sorted(list(self.char2phonemes.keys()))
        
        # bopomofo 到拼音
        with open(os.path.join(model_dir,'dict/bopomofo_to_pinyin_wo_tune_dict.json'),'r',encoding='utf-8') as fr:
            self.bopomofo_convert_dict = json.load(fr)
        # char bopomofo 字典
        with open(os.path.join(model_dir, 'dict/char_bopomofo_dict.json'),'r',encoding='utf-8') as fr:
            self.char_bopomofo_dict = json.load(fr)

        # 导入onnx 
        poly_onnx_path = os.path.join(model_dir,'g2pw.onnx')
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 2
        if DEVICE == 'cuda':
            self.session_g2pW = onnxruntime.InferenceSession(poly_onnx_path,sess_options=sess_options, providers=['CUDAExecutionProvider'])
        else:
            self.session_g2pW = onnxruntime.InferenceSession(poly_onnx_path,sess_options=sess_options, providers=['CPUExecutionProvider'])

        # warmup 
        input_1 = os.path.join(model_dir,'warmup/input_1.p')
        ort_inputs_1 = pickle.load(open(input_1,'rb'))
        probs = self.onnx_infer(ort_inputs_1)
        input_5 = os.path.join(model_dir,'warmup/input_5.p')
        ort_inputs_5 = pickle.load(open(input_5,'rb'))
        probs = self.onnx_infer(ort_inputs_5)
        input_8 = os.path.join(model_dir,'warmup/input_8.p')
        ort_inputs_8 = pickle.load(open(input_8,'rb'))
        probs = self.onnx_infer(ort_inputs_8)
        print("G2pw onnx loaded.")


    def get_phoneme_labels(self,polyphonic_chars: List[List[str]]
                           ) -> Tuple[List[str], Dict[str, List[int]]]:
        labels = sorted(list(set([phoneme for char, phoneme in polyphonic_chars])))
        char2phonemes = {}
        for char, phoneme in polyphonic_chars:
            if char not in char2phonemes:
                char2phonemes[char] = []
            char2phonemes[char].append(labels.index(phoneme))
        return labels, char2phonemes
    
    def poly_deal_berttoken(self,textlist):
    
        # consider cut text the textlist  not inclue [PAD]
        # include bert tokennizer and unk token
        textlist_tokenizer = []
        textlist_tokenizer.append(self.bert_dict["[CLS]"])
        for nn in textlist:
            # bert 英文为小写
            nn = nn.lower()
            if nn in self.bert_dict.keys():
                textlist_tokenizer.append(self.bert_dict[nn])
            else:
                textlist_tokenizer.append(self.bert_dict["[UNK]"])
    
        textlist_tokenizer.append(self.bert_dict["[SEP]"])
    
        return   textlist_tokenizer 
    
    def truncate_text(self,text_lists,query_lists,window_size):
    
        truncated_text_lists = []
        truncated_query_lists = []
        text_len_lists = [ len(ll) for ll in text_lists ]
        max_window_size = 30
        # text_lists 原始每个元素的数据长度为整个句子
        if window_size is None:
            if min(text_len_lists) <= max_window_size:
                window_size = min(text_len_lists)
            else:
                window_size = max_window_size
    
        # 如果数据长度少于窗长 
        if min(text_len_lists) <= window_size:
            truncated_text_lists = text_lists
            truncated_query_lists = query_lists
        else:
            for text,query_id in zip(text_lists,query_lists):
                query_start = query_id - window_size // 2
                query_end = query_id + window_size  //2
                if query_start <0:
                    start = 0
                    end = window_size
                else:
                    if query_end >= len(text):
                        end = len(text)
                        start = len(text) - window_size
                    else:
                        start = query_start 
                        end = query_end
    
                truncated_text = text[start:end]
                truncated_text_lists.append(truncated_text)
                truncated_query_id = query_id - start
                truncated_query_lists.append(truncated_query_id)
        return truncated_text_lists, truncated_query_lists


    def generate_poly_inputs(self,
                             sentence:List[str],
                             polyindex_lists):

        #
        use_mask = self.use_mask
        window_size = self.window_size
        if self.enable_non_tradional_chinese:
            translated_sentence = []
            for char in sentence:
                if char in self.s2t_dict:
                    translated_char = self.s2t_dict[char]
                else:
                    translated_char = char
                translated_sentence.append(translated_char)
            sentence = translated_sentence

        texts, query_ids, sent_ids,poly_chars = [], [], [],[]

        for i,char in enumerate(sentence):
            if  i in polyindex_lists:
                texts.append(sentence)
                query_ids.append(i)
                sent_ids.append(0)
                poly_chars.append(char)
    
        # 文本截取
        text_lists,query_lists = self.truncate_text(texts,query_ids,window_size)
        # 输入生成
        input_ids = []
        token_type_ids = []
        attention_mask = []
        phoneme_mask = []
        char_ids = []
        position_ids = []
        ort_inputs = {}
    
        for text,query_id in zip(text_lists,query_lists):
            input_id  = self.poly_deal_berttoken(text)
            token_type_id = list(np.zeros((len(input_id), ), dtype=int))
            attentionmask = list(np.ones((len(input_id), ), dtype=int))
            query_char = text[query_id]
            phonememask = [1 if i in self.char2phonemes[query_char] else 0 for i in range(len(self.labels))] 
            char_id = self.chars.index(query_char)
            position_id = query_id + 1 
            input_ids.append(input_id)
            token_type_ids.append(token_type_id)
            attention_mask.append(attentionmask)
            phoneme_mask.append(phonememask)
            char_ids.append(char_id)
            position_ids.append(position_id)
        ort_inputs = {
            'input_ids': np.array(input_ids).astype(np.int64),
            'token_type_ids': np.array(token_type_ids).astype(np.int64),
            'attention_mask': np.array(attention_mask).astype(np.int64),
            'phoneme_mask': np.array(phoneme_mask).astype(np.float32),
            'char_ids': np.array(char_ids).astype(np.int64),
            'position_ids': np.array(position_ids).astype(np.int64),
        }
        return ort_inputs,sent_ids,query_ids

    def onnx_infer(self,ort_inputs):

        probs = self.session_g2pW.run([], ort_inputs)[0]
        return probs

    def convert_bopomofo_to_pinyin(self,bopomofo: str) -> str:
        tone = bopomofo[-1]
        assert tone in '12345'
        component = self.bopomofo_convert_dict.get(bopomofo[:-1])
        if component:
            return component + tone
        else:
            print(f'Warning: "{bopomofo}" cannot convert to pinyin')
            return None
    

    def poly_post(self,probs,sent_ids,query_ids): 
        all_preds = []
        all_confidences = []
        preds = np.argmax(probs, axis=1).tolist()
        max_probs = []
        for index, arr in zip(preds, probs.tolist()):
            max_probs.append(arr[index])
        all_preds += [self.labels[pred] for pred in preds]
        all_confidences += max_probs
        results = {}
        for sent_id, query_id, pred in zip(sent_ids, query_ids, all_preds):
            results[query_id+1] = self.convert_bopomofo_to_pinyin(pred)
        return results
