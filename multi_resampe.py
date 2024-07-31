#!/usr/bin/env python
# coding: utf-8

# python multiprocess 


import os,sys
import subprocess
import argparse
import logging
import multiprocessing
from multiprocessing import Process
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description='multiprocess convert to  wav(8k,16bit,mono) using sox')
parser.add_argument('--num_process',type=int,help='multiprocess num')
parser.add_argument('--wavdir',type=str,help='wavdir inlcude .wav')
args = parser.parse_args()

def list_split(listtemp,n):
    ''' list split for multiprocess '''
    for i  in range(0,len(listtemp),n):
        yield listtemp[i:i+n]

def wavlist(wavdir):
    ''' Get wavdir include wavlist'''
    wavpath=[]
    for root,dirs,files in os.walk(wavdir):
        for file in files:
            filepath = os.path.join(root,file)
            if '.wav' in filepath:
                wavpath.append(filepath)
    return wavpath

def run_shell(subwavpath,soxtool,srcdir,outdir):
    ''' resample  8k 16bit mono  wav'''
    print ("run shell in {}".format(multiprocessing.current_process().name))
    for i in tqdm(range(len(subwavpath))):
        wavpath = subwavpath[i]
        #print (wavpath)
        outpath = wavpath.replace(srcdir,outdir)
        
        dstdir,wavname = os.path.split(outpath)
        if not os.path.exists(dstdir):
            os.makedirs(dstdir)
        cmd = "{} -t wav {} -c 1 -b 16 -r 8000 -t wav {}".format(soxtool,wavpath,outpath)
        res = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = res.communicate()
        if err.decode("utf-8") == "":
            pass
            #print ("Resample Successful:{}".format(wavpath))
        else:
            logging.error("{} Resample Error:{} output:{}".format(wavname,err,output.decode("utf-8")))
            #print ("wav:{} Resample error:{} output:{}".format(wavpath,err,output.decode("utf-8")))


if __name__ == "__main__":

    #[n,srcdir] = sys.argv[1:]
    n = args.num_process
    srcdir = args.wavdir
    srcdir = os.path.abspath(srcdir)
    n = int(n) # multiprocess num
    # log
    logging.basicConfig(filename='resample.log',format='%(asctime)s:%(message)s',level=logging.INFO) 
    logging.info("start")
    logging.info("resample {}".format(srcdir))
    # path
    start = time.time()
    soxtool=r'/usr/bin/sox'
    wavpath = wavlist(srcdir)
    outdir = srcdir+'_8k'
    # split list
    subnum = int(len(wavpath)/n) # 
    print (subnum,len(wavpath)) 
    temp = list_split(wavpath,subnum) # store list
    plist=[]
    for i in temp:
        p = Process(target=run_shell,args=(i,soxtool,srcdir,outdir,))
        plist.append(p)
        p.start()
    for p in plist:
        p.join()

    print ("Process end")
    end = time.time()
    dur = end - start 
    print ("process {} runtime is {}".format(n,dur))
    logging.info("process {} runtime is {}".format(n,dur))


