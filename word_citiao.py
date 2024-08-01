import requests
import os,sys
import re
from tqdm import tqdm 

# 功能：获取对应网址的网页
class HtmlDownloader(object):
    def download(self, base_url,word):
        headers_pc = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36',
                   'Accept': 'text / html, application / xhtml + xml, application / xml;q = 0.9,image/webp, * / *;q = 0.8'}
        url = '{}/{}'.format(base_url,word)
        response = requests.get(url,headers=headers_pc,timeout=60)
        if response.status_code != 200:
            return None
        return response.content


if __name__ == "__main__":

    wordfile,infile,nofile = sys.argv[1:]

    word_lines = [ line.strip() for line in open(wordfile,'r',encoding="utf-8") ]
    downloader=HtmlDownloader()

    inwords = []
    nowords = []

    base_url = 'https://baike.baidu.com/item'
    with open(infile,'w+',encoding="utf-8") as fi,open(nofile,'w+',encoding="utf-8") as fn:
        for i in tqdm(range(0,len(word_lines))):
            word = word_lines[i].split()[0]
            try:
                htm=downloader.download(base_url,word).decode('utf-8')
            except Exception as e:
                print (e)
            else:
                if re.search(r'https://baike.baidu.com/error.html',htm):
                    fn.write('{}\n'.format(word))
                else:
                    fi.write('{}\n'.format(word))
    
