!/usr/bin/env python3 

import os,sys,re

def generate_pinyin(inputfile):

    phrases_dict = {}

    alllines = [ line.strip() for line in open(inputfile,'r',encoding="utf-8")]

    for line in alllines:
        word = line.split()[0]
        pinyins = [ [pp] for pp in line.split()[1:] ]
        phrases_dict[word] = pinyins

    return phrases_dict

def read_dict(dictfile):

    word2phones = {}
    with open(dictfile,'r',encoding="utf-8") as fp:
        for line in fp.readlines():
            line = line.strip()
            word = line.split()[0]
            phonelists = line.split()[1:]
            word2phones[word] = phonelists

    return word2phones

def sentence_tokenizer(sentence):
    # For 中文 中英混
    list1 = re.split(r'([a-zA-Z]+)',sentence)
    selists =[]
    for ss in list1:
        if re.match(r'[A-Za-z]+$',ss):
            selists.append(ss)
        else:
            for sss in ss:
                if sss.strip():
                    selists.append(sss)
    return selists

def sentence_split(sentence_lists,max_len=100):
    
    new_sentence_lists = [] 
    nums = len(sentence_lists) 

    s_sum = 0 
    tmp_ss = "" 

    for i,ss in enumerate(sentence_lists):
        ss_token = sentence_tokenizer(ss)
        ss_len = len(ss_token)
        #print (ss,ss_len)

        if ss_len < max_len:
            if s_sum+ss_len  <= max_len:
                tmp_ss += ss
                s_sum += ss_len
            else:
                new_sentence_lists.append(tmp_ss)
                tmp_ss = ss
                s_sum = ss_len
            if i == (nums -1):
                if len(tmp_ss) >0:
                    new_sentence_lists.append(tmp_ss)

        else:
            if len(tmp_ss) >0:
                new_sentence_lists.append(tmp_ss)
            tmp_ss =""
            iternum = ss_len//max_len 
            for j in range(iternum):
                start = j*max_len
                end = (j+1)*max_len
                inner_ss_token = [ss_token[j]+" " if re.match(r'[A-Za-z]',ss_token[j]) else ss_token[j]  for j in range(start,end) ]
                inner_sentence = ''.join(inner_ss_token)
                new_sentence_lists.append(inner_sentence)
            tail_ss_token = [ss_token[j]+" " if re.match(r'[A-Za-z]',ss_token[j]) else ss_token[j]  for j in range(iternum*max_len,ss_len) ]
            tmp_ss = ''.join(tail_ss_token)
            if i == (nums -1):
                if len(tmp_ss) > 0:
                    new_sentence_lists.append(tmp_ss)

    return new_sentence_lists
def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result
def get_poly_index(part_word_count_lists,polyword_dict):

    # 获取到多音字对应的下标
    poly_index_lists = []
    for i, ww in enumerate(part_word_count_lists[1:-1]):
        if re.match(r'[\u4e00-\u9fa5]',ww[0]):
            word,phonenums,pinyin_flag = ww
            if word  in polyword_dict.keys() and  pinyin_flag == '0':
                poly_index_lists.append(i)
    return  poly_index_lists
