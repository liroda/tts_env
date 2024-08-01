import os,sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
 
from bs4 import BeautifulSoup
from tqdm import tqdm 


def  get_polycizu(url):
    # 打开网页
    driver.get(url)
    # 找到所有需要点击的元素，加载等待后再点击
    try:
        elements = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.XPATH, "//button[@class='btn btn-outline-danger btn-round']")))
    # driver.execute_script("arguments[0].click();", element)
        for element in elements:
            driver.execute_script("arguments[0].click();", element)
    except:
        print("元素未在30秒内加载出来")
    
      # print(len(element), element[2].get_attribute('innerHTML'))
    word2pinyin = {}

    try:
        WebDriverWait(driver, 60).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.ci-list > ul >li')))
        
        contents = driver.find_elements(By.CSS_SELECTOR, ".ci-list > ul >li")
        print(len(contents))
        
        for item in contents:
            html = item.get_property('innerHTML')
            li = BeautifulSoup(html, 'html.parser')
            a =  li.select('.stretched-link')[0]
            p = li.select('.pinyin')[0]
            word = a.attrs.get('title')
            pinyin = p.text
            word2pinyin[word] = pinyin
    except:
        print ('元素未在60秒内加载出来')
   
    return word2pinyin
    
    
print('开始加载网页内容')
 # 启动浏览器驱动
driver = webdriver.Chrome()

if __name__ == "__main__":

    polyfile,outdir = sys.argv[1:]

    print('开始加载网页内容')
     # 启动浏览器驱动
    driver = webdriver.Chrome()

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    poly_words = [ line.strip() for line in open(polyfile,'r',encoding="utf-8")]

    base_url = 'http://hanyuguoxue.com/zuci'
    
    for i in tqdm(range(len(poly_words))):
        word = poly_words[i]
        poly_url = '{}/zi-{}'.format(base_url,ord(word))
        word2pinyin = get_polycizu(poly_url)
        outfile = '{}/polyword_{}'.format(outdir,word)
        with open(outfile,'w',encoding="utf-8") as fw:
            for word,pinyin in word2pinyin.items():
                fw.write('{}\t{}\n'.format(word,pinyin))
    driver.quit()
