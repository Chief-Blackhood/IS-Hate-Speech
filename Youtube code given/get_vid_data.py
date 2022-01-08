from selenium import webdriver 
import pandas as pd 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import numpy as np
import sys

options = webdriver.ChromeOptions()
options.add_argument("start-maximized")
options.add_argument("disable-infobars")
options.add_argument("disable-extensions")
driver = webdriver.Chrome(chrome_options=options)

news_channels = ["https://www.youtube.com/playlist?list=PL8QBkS_wk32ULw1VExquey4rf2eOL65OS"]
ids = []
descriptions = []
titles = []
count = 0

for num, page in enumerate(news_channels):
    links = []
    driver.get(page)
    htmlbody = driver.execute_script("return document.body.innerHTML;")
    f=open("docBodyListPage.txt","w") 
    f.write(str(htmlbody)+"\n")
    f.close()
    elem = driver.find_element_by_tag_name("body")
    #change number of pagedowns for more clips
    no_of_pagedowns = int(sys.argv[1])
    for num in range(no_of_pagedowns):
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(1.0)
    #     no_of_pagedowns-=1

    videoURLs = driver.find_elements_by_xpath('//*[@id="content"]/a')
    print("Found "+str(len(videoURLs))+" videos")
    for i in (videoURLs):
        try:
            links.append(i.get_attribute('href').split('&')[0])
            print("Video URL: "+i.get_attribute('href').split('&')[0])
        except:
            pass
        

    wait = WebDriverWait(driver, 100)
    for i in links:
#        try:
        print(str(count))
        driver.get(i)
        time.sleep(2)
        htmlbody = driver.execute_script("return document.body.innerHTML;")
        f=open("videoDescPage"+str(count)+".txt","w", encoding="utf-8") 
        f.write(str(htmlbody)+"\n")
        f.close()
        duration = driver.find_element_by_class_name('ytp-time-duration').text
        print(duration)
        try:
            timed = int(duration.split(':')[0])*60 + int(duration.split(':')[1])
        except:
            timed=100
        if timed > 250:
            print('skipped ' + duration)
        else:
            titles.append(wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"h1.title yt-formatted-string"))).text)
            descriptions.append(wait.until(EC.presence_of_element_located((By.CSS_SELECTOR,"div#description yt-formatted-string"))).text)
            ids.append(i)
            print(duration)
            print(titles[-1])
            print(ids[-1])
        count += 1
        if count > 5:
            break
#        except:
#            print("Could not scrape " + i)
    np.save('links'+str(num)+'.npy', np.array(links))

df = pd.DataFrame(data = {'ids':ids, 'titles':titles, 'descriptions':descriptions})
df.to_csv('video_data.csv')
