!/usr/bin/env python3
import os,re
import numpy as np

import jieba.posseg as psg
import jieba


from pypinyin import lazy_pinyin
from pypinyin import load_phrases_dict
from pypinyin import load_single_dict
from pypinyin import Style
from pypinyin_dict.phrase_pinyin_data import large_pinyin

from g2p_en import G2p

from zh_normalization.text_normlization import TextNormalizer
from tone_sandhi import ToneSandhi
from front_utils import generate_pinyin,read_dict,sentence_tokenizer,sentence_split,intersperse
from map_en2cn import map_lexicon,fix_en,en_initials_finals

from collections import defaultdict
import traceback
import copy
from text_rule import poly_change
from poly_infer_onnx import G2PW_Process
from pypinyin.constants import PHRASES_DICT
from front_utils import get_poly_index
import pickle


class VITS_TextFront:
    def __init__(self,phoneid_map_path=None,
                      jieba_userdict_path=None,
                      poly_dict_path=None,
                      bert_dict_path=None,
                      symbol_pinyin_path=None,
                      pinyin2phone_path=None,
                      cmudict_path=None,
                      polyword_dict_path=None,
                      polymodel_dir=None,
                      combine=True):

        self.text_normalizer = TextNormalizer()
        self.tone_modifier = ToneSandhi()

        self.phoneid_map_path = phoneid_map_path
        self.jieba_userdict_path = jieba_userdict_path
        self.poly_dict_path = poly_dict_path
        self.bert_dict_path = bert_dict_path
        self.symbol_pinyin_path = symbol_pinyin_path
        self.pinyin2phone_path = pinyin2phone_path
        self.cmudict_path = cmudict_path
        self.polyword_dict_path = polyword_dict_path
        self.polymodel_dir = polymodel_dir

   
        # bert 
        alllines_bertdict = [ line.strip()  for line in open(self.bert_dict_path,'r',encoding="utf-8") ]
        self.bertdict = {line:i   for i,line in enumerate(alllines_bertdict) }

        #jieba
        jieba.load_userdict(self.jieba_userdict_path)

         # pinyin
        large_pinyin.load()
        # polyphones
        self.poly_phrases_dict = generate_pinyin(self.poly_dict_path)
        load_phrases_dict(self.poly_phrases_dict)

        # 调整字的拼音顺序
        load_single_dict({ord(u'地'): u'de,di4'})
        load_single_dict({ord(u'幢'): u'zhuang4,chuang2'})


        # 特殊符号发音
        self.symbol_pinyin = generate_pinyin(self.symbol_pinyin_path)

        # pinyin - phone
        self.pinyin_dict = read_dict(self.pinyin2phone_path)

        # english
        self.en_g2p = G2p()
        self.mapen2cn = map_lexicon()
        self.fix_en2cn = fix_en()
        # cmu dict
        self.cmu_dict = read_dict(self.cmudict_path)
        
        # convert phone to phone_id(int)
        self.phoneid_map_path = phoneid_map_path
        self.phone2id = { line.split()[0]:line.strip().split()[-1]  for line in open(self.phoneid_map_path,'r',encoding="utf-8")}

        self.combine = combine
        
        # 多音字模型
        self.Polymodel = True
        self.polyword_dict = { line.strip():1  for line in open(self.polyword_dict_path,'r',encoding="utf-8") }
        self.g2pw_process = G2PW_Process(model_dir=self.polymodel_dir)
        

    def deal_berttoken(self,selists):
        # include bert tokennizer and unk token
        selists_tokenizer = []
        selists_tokenizer.append(self.bertdict["[PAD]"])
        for nn in selists[1:-1]:
            # bert 英文为小写
            nn = nn.lower()
            if nn in self.bertdict.keys():
                selists_tokenizer.append(self.bertdict[nn])
            else:
                selists_tokenizer.append(self.bertdict["[UNK]"])

        selists_tokenizer.append(self.bertdict["[PAD]"])

        return   selists_tokenizer

    def pinyin_to_phone(self,pinyin,pinyin_dict):

        phone = []

        try:
            if not pinyin[-1].isdigit():
                pinyin += "5"
            if pinyin[:-1] in pinyin_dict:
                tone = pinyin[-1]
                a = pinyin[:-1]
                a1, a2 = pinyin_dict[a]
                phone = [a1, a2 + tone]
        except Exception as e:
            print ("pinyin_to_phone warning",e) 
    
        return phone
    def english_to_phone(self,word,map_to_cn=False):
        # cmu dict 英文字母为大写
        word = word.upper()
        if word in self.fix_en2cn.keys():
            phones  = self.fix_en2cn[word]
        elif word in self.cmu_dict.keys():
            en_phones = copy.deepcopy(self.cmu_dict[word])
            if len(word) == 1:
                en_phones.append('sp')
            if map_to_cn:
                phones = en_initials_finals(en_phones,self.mapen2cn)
            else:
                # 训练包含英文phone,不进行映射
                phones = en_phones
        else:
            en_phones = self.en_g2p(word)
            if map_to_cn:
                phones = en_initials_finals(en_phones,self.mapen2cn)
            else:
                phones = en_phones
        if len(word) >1:
            return ['^'] + phones
        else:
            return phones
    def special_pinyin(self,key):
        if key in self.symbol_pinyin.keys():
            return self.symbol_pinyin[key][0]
        else:
            return key
    def get_sentence_phonemes(self,sentence):

        textstr = ' '.join(re.split(r'([a-zA-Z]+)',sentence))
        seg_cut = psg.lcut(poly_change(textstr))
        try:
            seg_cut = self.tone_modifier.pre_merge_for_modify(seg_cut)
        except Exception as e:
            print ("seg merge warning",e)

        seg_cut = [ (word,pos)  for word,pos in seg_cut if word.strip()]
        #print ('seg_cut is {}'.format(seg_cut))
        # 存储单个字或者英语及其对应的音素个数，中文一般是2个，sp 代表1个
        word_count_lists = []
        # 存储phone
        phone_items = []
        word_count_lists.append(('[PAD]',1))
        phone_items.append('sil')

        for i,(word,pos) in enumerate(seg_cut):
            wordnum_lists,wordphone_lists = self.get_word_phonemes(word,pos)
            try:
                if word == "为" and  seg_cut[i-1][0] in  ["更新","变更"]:
                    wordphone_lists[-1] = wordphone_lists[-1][:-1] +"2"
            except Exception as e:
                print ("wei warning ",e)
            word_count_lists.extend(wordnum_lists)
            phone_items.extend(wordphone_lists)

        word_count_lists.append(('[PAD]',1))
        phone_items.append('sil')

        return word_count_lists,phone_items 
    def get_word_phonemes(self,word,pos):

        wordphone_lists = []
        wordnum_lists = []

        # 全英文
        if re.match(r'[A-Za-z]+$',word):
            wordphones = self.english_to_phone(word)
            wordnum_lists.append((word,len(wordphones)))
            wordphone_lists.extend(wordphones)
        else:
            pinyins = lazy_pinyin(word,neutral_tone_with_five=True,style=Style.TONE3,errors = lambda x:self.special_pinyin(x))
            #print (word,pinyins)
            if len(word) == 1:
                flag = '0'
            else:
                if word in PHRASES_DICT.keys():
                    flag = '1'
                else:
                    flag = '0'
            
            for j,ww in enumerate(word):
                sing_phones = []
                try:
                    if re.match(r'[\u4e00-\u9fa5]+',ww):
                        sing_phones = self.pinyin_to_phone(pinyins[j],self.pinyin_dict)
                    elif ww in self.symbol_pinyin.keys():
                        sing_phones = self.pinyin_to_phone(pinyins[j],self.pinyin_dict)
                    else:
                        sing_phones = ['sp']
                except Exception as e:
                    print ("pinyin2phone  warning:",traceback.format_exc())
                else:
                    wordnum_lists.append((ww,len(sing_phones),flag))
                    wordphone_lists.extend(sing_phones) 
            len_phones = len(wordphone_lists)
            #if len_phones >= 4 and re.match(r'[\u4e00-\u9fa5]+$',word):
            if  (len_phones % 2) == 0 and re.match(r'[\u4e00-\u9fa5]+$',word):
                sub_initials = wordphone_lists[0:len_phones:2]
                sub_finals = wordphone_lists[1:len_phones:2]
                try:
                    sub_finals = self.tone_modifier.modified_tone(word, pos,sub_finals)
                except:
                    print ("tone warning")
                else:
                    modify_wordphone_lists = []
                    for ii,ee in zip(sub_initials,sub_finals):
                        modify_wordphone_lists.append(ii)
                        modify_wordphone_lists.append(ee)
                    wordphone_lists = modify_wordphone_lists

        return wordnum_lists,wordphone_lists
    def poly_run(self,part_word_count_lists,part_phone_items,polyindex_lists):

        print (part_phone_items)
        polyid2pinyin = {}
        try:
            part_polytext_tokenizer = [ ww[0]  for ww in part_word_count_lists[1:-1] ]
            #print (part_polytext_tokenizer)
            ort_inputs,sent_ids,query_ids = self.g2pw_process.generate_poly_inputs(part_polytext_tokenizer,polyindex_lists)
            probs = self.g2pw_process.onnx_infer(ort_inputs)
            polyid2pinyin  = self.g2pw_process.poly_post(probs,sent_ids,query_ids)
            print (polyid2pinyin)
        except Exception as e:
            print ("poly error warning ",e)
        else:
            # 多音字替换pingyin 
            if len(list(polyid2pinyin.keys())) > 0:
                new_part_word_count_lists = []
                new_part_phone_items = []
                for i, word_phonelens in enumerate(part_word_count_lists):
                    if i in polyid2pinyin.keys():
                        pinyin_value = polyid2pinyin[i]
                        poly_phones = self.pinyin_to_phone(pinyin_value,self.pinyin_dict)
                        new_part_word_count_lists.append((part_word_count_lists[i][0],len(poly_phones),'0'))
                        new_part_phone_items.extend(poly_phones)
                    else:
                        new_part_word_count_lists.append(part_word_count_lists[i])
                        start_lists = [ part_word_count_lists[j][1]   for j in range(0,i) ]
                        start = sum(start_lists)
                        end = start + part_word_count_lists[i][1]
                        new_part_phone_items.extend(part_phone_items[start:end])
                if len(new_part_phone_items) == len(part_phone_items):
                    part_word_count_lists = new_part_word_count_lists
                    part_phone_items  = new_part_phone_items
        #print (part_word_count_lists)
        #print (part_phone_items)
        return part_word_count_lists,part_phone_items

    def  get_front_inputids_embeds(self,sentence):

         
        # 文本正则化
        #print ("*****front begin*****")
        sentence = self.text_normalizer.normalize(sentence)
        #print ("sentence is {}".format(sentence))

         # 控制输入文本的长度
        try:
            sentence  = sentence_split(sentence)
            #print ("new split  is {}".format(sentence))
        except Exception as e:
            print ("split warning:",traceback.format_exc())
            #print ("split warning:",traceback.format_exc())


        result = []
        input_ids = []
        bert_input_ids = []
        bert_length = []

        for i,part_ss in enumerate(sentence):

            if re.search(r'[\u4e00-\u9fa5a-zA-Z](\s)?$',part_ss):
                part_ss = part_ss +"。"

            part_word_count_lists,part_phone_items = self.get_sentence_phonemes(part_ss)
            #print ('part_word_count_lists is {}'.format(part_word_count_lists))
            # 多音字模型
            if self.Polymodel:
                poly_index_lists = get_poly_index(part_word_count_lists,self.polyword_dict)
                if len(poly_index_lists) > 0:
                    part_word_count_lists,part_phone_items = self.poly_run(part_word_count_lists,part_phone_items,poly_index_lists)


            # phone id for word
            part_real_phonemes =[ phn if phn in self.phone2id.keys() else "sp" for phn in part_phone_items]
            part_input_ids = [ int(self.phone2id[pp])  for pp in part_real_phonemes ]
            input_ids.extend(part_input_ids)

            # bert 拆分为token
            part_text_tokenizer = [ ww[0] for ww in part_word_count_lists ]  
            part_bert_length = [ww[1] for ww in part_word_count_lists ]
            part_bert_input_ids = self.deal_berttoken(part_text_tokenizer)
            bert_input_ids.extend(part_bert_input_ids)
            bert_length.extend(part_bert_length)



            if not self.combine:
                x_tst_np = np.array(part_input_ids,dtype=np.int64).reshape(1,-1)
                x_tst_lengths_np = np.array([len(part_input_ids)],dtype=np.int64)

                x_tst_bert_input_ids = np.array(part_bert_input_ids)
                x_tst_bert_length = np.array(part_bert_length)
                
        if self.combine:
            max_phone_length = 502
            max_word_length = 512
            #print ('input_ids length:{}  word_length:{}'.format(len(input_ids),len(bert_input_ids)))
            if len(input_ids) > max_phone_length:
                phone_length = max_phone_length
            else:
                phone_length = len(input_ids)

            if len(bert_input_ids) > max_word_length:
                word_length = max_word_length
            else:
                word_length = len(bert_input_ids)

            #print ('input_ids length:{}  word_length:{}'.format(phone_length,word_length))
            x_tst_np = np.array(input_ids[0:phone_length],dtype=np.int64).reshape(1,-1)
            x_tst_lengths_np = np.array([phone_length],dtype=np.int64)

            x_tst_bert_input_ids = np.array(bert_input_ids[0:word_length],dtype=np.int32).reshape(1,-1)
            x_tst_bert_length = np.array([bert_length[0:word_length]],dtype=np.int32)

            result.append({'x':x_tst_np.tolist(),'x_length':x_tst_lengths_np.tolist(),'word_input_ids':x_tst_bert_input_ids.tolist(),'expand_length':x_tst_bert_length.tolist()})
        return result
