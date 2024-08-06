#!/usr/bin/env python3
import os,sys,re


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
   selists.insert(0,'[PAD]')
   selists.append('[PAD]')
   return selists

def deal_untoken_oov(selists,worddict):
    selists_b = []
    for nn in selists:
        nn = nn.lower()
        if nn in worddict.keys() or nn in [ '[PAD]']:
            selists_b.append(nn)
        else:
            selists_b.append("[UNK]")

    return selists_b

def pinyins_split(pinyinstr):

    pinyinstr = pinyinstr.replace(r'.','')
    pinyinlists = re.split(r'[\/]',pinyinstr)
    pinyins =[]
    for p in pinyinlists:
        if re.match('[\s]*[a-z]',p):
            pinyins.extend(p.split())
        elif re.match('[\s]*[A-Z]+',p):
            pinyins.append(p.split())
    return pinyins

def get_align(selists,pinyins,pinyin_dict):

    count =[]
    phones = []
    j = 0
    #print (selists)
    for ww in selists:
        if re.match(r'[\u4e00-\u9fa5]',ww):
            count.append(2)
            pinyin = pinyins[j]
            tone = pinyin[-1]
            a = pinyin[:-1]
            #print (ww,a)
            a1, a2 = pinyin_dict[a]
            phones.append([a1, a2 + tone])
            j +=1
        elif re.match(r'[A-Za-z]',ww):
            count.append(len(pinyins[j]))
            phones.append(pinyins[j])
            j += 1
        elif re.search(r'\[PAD\]',ww):
            count.append(1)
            phones.append('sil')
        else:
            count.append(1)
            phones.append('sp')
        #print (ww,phones[-1])
    phones_item = []
    for pp in phones:
        if isinstance(pp,str):
            phones_item.append(pp)
        elif isinstance(pp,list):
            phones_item.extend(pp)
    return count,phones_item
#def read_dict(dictfile):
    
#    symbols = [ line.strip().split()[0] for line in open(dictfile,'r',encoding="utf-8") ]
#    return symbols
#symbols = read_dict(r'phone_mix.txt')
