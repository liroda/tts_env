#!/usr/bin/env python3
import re
import time
def map_lexicon():

    map_en2cn ={'AA': 'a', 
               'AE': 'an', 
               'AH': 'e', 
               'AO': 'ang', 
               'AW': 'ao', 
               'AY': 'ai', 
               'B': 'b', 
               'CH': 'ch', 
               'D': 'd', 
               'DH': 'r', 
               'EH': 'en', 
               'ER': 'er', 
               'EY': 'ei', 
               'F': 'f', 
               'G': 'g', 
               'HH': 'h', 
               'IH': 'i', 
               'IY': 'ii', 
               'JH': 'z', 
               'K': 'k', 
               'L': 'l', 
               'M': 'm', 
               'N': 'n', 
               'NG': 'n,g', 
               'OW': 'ou', 
               'OY': 'ang,ai', 
               'P': 'p', 
               'R': 'r', 
               'S': 's', 
               'SH': 'sh', 
               'T': 't', 
               'TH': 'sh', 
               'UH': 'u', 
               'UW': 'u', 
               'V': 'v1', 
               'W': 'u1', 
               'Y': '^', 
               'Z': 'z', 
               'ZH': 'zh'}

     
    return map_en2cn

def fix_en():
    fix_en2cn = {}
    fix_en2cn['A'] = (['ei1'])
    fix_en2cn['B'] = (['b','i4'])
    fix_en2cn['C'] = (['s','ei1'])
    fix_en2cn['D'] = (['d','i4'])
    fix_en2cn['E'] = (['^','i1'])
    fix_en2cn['F'] = (['an2','f','u1'])
    #fix_en2cn['G'] = (['j','i1'])
    fix_en2cn['H'] = (['^','ai1','ch'])
    fix_en2cn['I'] = (['^','a4','ai1'])
    #fix_en2cn['J'] = (['j','ei1'])
    fix_en2cn['K'] = (['k','ei1'])
    fix_en2cn['L'] = (['^','an1','l'])
    fix_en2cn['M'] = (['^','an1','m'])
    fix_en2cn['N'] = (['^','en1'])
    fix_en2cn['O'] = (['^','ou1'])
    fix_en2cn['P'] = (['p','i1'])
    fix_en2cn['Q'] = (['k','ou1'])
    #fix_en2cn['R'] = (['a1','er2'])
    fix_en2cn['S'] = (['an2','s','ii1'])
    fix_en2cn['T'] = (['t','i1'])
    fix_en2cn['U'] = (['^','iou1'])
    fix_en2cn['V'] = (['^','uei1'])
    #fix_en2cn['W'] = (['d','a1','b','u1','l','iou2'])
    #fix_en2cn['X'] = (['ai1','k','e4','s','ii5'])
    #fix_en2cn['Y'] = (['^','uan1'])
    fix_en2cn['Z'] = (['r','ei4'])
    fix_en2cn['SF'] = (['an2','s','an2','f','u1'])
    #fix_en2cn['APPLE'] = (['an1','p','ao5','l'])
    #fix_en2cn['IPHONE'] = (['ai1', 'f','ou5', 'n'])

    return fix_en2cn


def en_initials_finals(en_word,en2cn):

    orig_initials = []
    orig_finals = []

    cn_start = time.time()
    for ww in en_word:
       #print (ww)
       real_en = re.sub(r'[0-9]','',ww)
       num = re.findall(r'[0-9]',ww)
       real_cn_list = en2cn[real_en].split(',')
       for cc in real_cn_list:
           if num in [ ["1"],["2"],["3"],["4"]]:
               orig_initials.append( cc+num[0] )
           elif num in [["0"]]:
               orig_initials.append( cc+"5" )
           else:
               orig_initials.append(cc)
           orig_finals.append('')
    #print ("cn_dur is {:.4f} ms".format((time.time() - cn_start)*1000))
    return orig_initials 
