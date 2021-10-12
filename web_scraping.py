# from UTILITY_COMMON_OGILVY import *

import requests
import pandas as pd
import numpy as np
import time
import lxml
import sys
import datetime
import ast
import traceback
import logging
import jieba
import jieba.analyse
import jieba.posseg as pseg
import json
import os
import inspect
import traceback
import glob
import pymsteams
import re
import random
import pprint
import gc
import urllib3
import pprint

from pytz import timezone
from bs4 import BeautifulSoup
from selenium import webdriver
from urllib.request import urlopen, Request
from dateutil.relativedelta import relativedelta
from selenium.webdriver.common.keys import Keys
from collections import Counter
from webdriver_manager.chrome import ChromeDriverManager
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# from selenium.webdriver import Chrome

####################    DISABLE ANY PYTHON WARNINGS VIA THE PYTHONWARNINGS    ####################
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

#######################################################     USER      #######################################################
LOAD_MORE_CLICK_NUM = 3  # Default better to be 100 in order to cover full day shortcontent article (If >100, error: list index out of range)
TOP_KEYWORDS_NUM_PER_ARTICLE = 20  # TF-IDF Top N Keyword extraction per article
TOP_KEYWORDS_NUM_SUMMARY = 50  # Summary from all articles
PROGRAM_ABSOLUTE_PATH = '/Users/lapkeitang/Documents/PycharmProjects/Demo/'
EXPORT_ABSOLUTE_PATH = PROGRAM_ABSOLUTE_PATH

##########   FILTER (OPTIONAL)   ##########
FILTER = True  # Filter is effective only if FILTER = True
FILTER_TYPE = 'D'  # 'D' = Filter by a single date, 'M' = Filter by a single month

# FILTER_TYPE = 'D' i.e. Day
DAY_BACKWARD = 0  # 0 = Today, 1 = Yesterday, 2 = day before yesterday , 3 = and so on ...

# FILTER_TYPE = 'M' i.e. Month
FILTER_MONTH = datetime.datetime.now().month
# FILTER_MONTH = 7    # Doesn't matter if it is in 'str' or 'int'

#######################################################     GENERAL      #######################################################
# os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

# now = datetime.datetime.now() + relativedelta(hours=8)
now = datetime.datetime.now()
# now = datetime.datetime.now(timezone('Hongkong'))
# now = datetime.datetime.now(timezone('Greenwich'))
TODAY_STRING = now.strftime("%Y%m%d")
TODAY_TIME_STRING = now.strftime("%Y%m%d%H%M%S")

FILTER_DATE_STRING = (now + relativedelta(days=-DAY_BACKWARD)).strftime("%Y%m%d")

CSV_PATH = EXPORT_ABSOLUTE_PATH + 'csv/'
CSV_TODAY_PATH = CSV_PATH + TODAY_STRING + '/'

LOG_PATH = EXPORT_ABSOLUTE_PATH + 'log/'
LOG_TODAY_PATH = LOG_PATH + TODAY_STRING + '/'

#   TIMEZONE
Greenwich = timezone('Etc/Greenwich')
# London = timezone('Europe/London')
# GMT_plus_8 = timezone('Etc/GMT-8')
hk = timezone('Hongkong')


def title(string='', length=150, symbol='-'):
    if len(string) > 0:
        string = ' ' + string + ' '
    print('\n{0:{1}^{2}}'.format(string, symbol, length))


def create_directory(directory):
    if not os.path.exists(directory):
        print("Creating directory: {}".format(directory))
        os.makedirs(directory)


def get_time_elapsed(startTime, is_print=True):
    time_string = time.strftime('%H:%M:%S', time.gmtime(time.time() - startTime))
    if is_print:
        print('\nTime cost - {}'.format(time_string))
    return time_string


def top_k(iterable, k=10):
    """The counter.most_common([k]) method works
    in the following way:
    Counter('abracadabra').most_common(3)
    [('a', 5), ('r', 2), ('b', 2)]
    """
    c = Counter(iterable)
    # most_common = [key for key, val in c.most_common(k)]
    df = pd.DataFrame(c.most_common(k), columns=['KEYWORD', 'ARTICLE COUNT'])
    for k, v in c.most_common(k):
        print('{0} - article count(S): {1}'.format(k, v))
    print('\n')
    return df


def get_now(timezone=timezone('Hongkong')):
    now = datetime.datetime.now(timezone)
    now_string = now.strftime("%Y%m%d")
    now_time_string = now.strftime("%Y%m%d%H%M%S")
    now_time_string_exact = now.strftime("%Y%m%d%H%M%S.%f")
    now = datetime.datetime.strptime(now_time_string_exact, '%Y%m%d%H%M%S.%f')
    return now, now_string, now_time_string, now_time_string_exact


#######################################################     LOG   #######################################################
class Logger(object):
    def __init__(self, path, filename, filetype):
        self.path = path
        self.filename = filename
        self.filetype = filetype
        self.terminal = sys.stdout
        create_directory(path)
        self.log = open(path + filename + '.' + filetype, "a", encoding='utf-8')
        print("Stdout is save as {0}.{1} in path: {2}".format(filename, filetype, path))

    def __str__(self):
        return "Path: {0}, Filename: {1}, Filetype: {2}".format(self.path, self.filename, self.filetype)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


MODULE_NAME = os.path.basename(inspect.getfile(inspect.stack()[-1][0])).replace('.py', '')
sys.stdout = Logger(path=LOG_TODAY_PATH, filename='{0}_{1}'.format(TODAY_TIME_STRING, MODULE_NAME), filetype='log')
title(string=MODULE_NAME)

#######################################################     PROGRAM START   #######################################################
startTime = time.time()

#####   GET SOUP (URL_SOUP) FROM BROWSER   #####
option = webdriver.ChromeOptions()
option.add_argument("--incognito")
option.add_experimental_option('excludeSwitches', ['enable-logging'])  # To shut up that USB error
# browser = webdriver.Chrome(executable_path = 'C:/Users/nathan.tang/PycharmProjects/ML/chromedriver/chromedriver91.exe', options=option)
# browser = webdriver.Chrome(executable_path = PROGRAM_ABSOLUTE_PATH + 'chromedriver/chromedriver91.exe', options=option)
browser = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=option)
browser.implicitly_wait(30)
base_uri = 'https://bihu.com/hot'
browser.get(base_uri)
browser.maximize_window()

# main_soup = BeautifulSoup(browser.page_source, "html.parser")  # 'lxml'
# some_list = main_soup.find_all('div', attrs={'class': 'ant-tabs-tab-btn', 'aria-controls': 'rc-tabs-0-panel-SHORT'})
# print(some_list)

#   CLICK 微文
short_tab = browser.find_element_by_id(id_='rc-tabs-0-tab-SHORT')  # browser.find_elements_by_class_name("ant-tabs-tab-btn")
browser.execute_script("arguments[0].click();", short_tab)
time.sleep(random.uniform(1, 2))

#   CLICK 最新
latest_tab = browser.find_element_by_xpath('//button[text()="最新"]')
browser.execute_script("arguments[0].click();", latest_tab)
time.sleep(random.uniform(1, 2))

#   CLICK 加载更多
for _ in range(LOAD_MORE_CLICK_NUM):
    load_more_tab = browser.find_elements_by_xpath("//button[@class='jsx-1161939275 load-more'][.='加载更多']")[-1]
    browser.execute_script("arguments[0].click();", load_more_tab)
    time.sleep(random.uniform(1, 2))

# url_element_list = browser.find_elements_by_xpath("(//div[contains(@href, '')])")
url_soup = BeautifulSoup(browser.page_source, "html.parser")  # 'lxml'
browser.quit()

#####   GET DETAIL (DF_SHORTCONTENT) FROM SOUP (URL_SOUP)   #####
title(string="GET DETAIL - DF_SHORTCONTENT")

shortcontent_list = []
skip_count = 0
# url_list = url_soup.find_all('div', attrs={'class': 'short-content'})     # For checking only
url_list = ['https://bihu.com' + x['href'] for x in url_soup.find_all('a', href=True) if 'shortcontent' in x['href']]
if not not url_list:
    for url in url_list:
        shortcontent_dict = {}

        try:
            # print("URL: {}".format(url))
            res = requests.get(url)
        except:
            # print("Bad Handshake Error URL: {}".format(url))
            res = requests.get(url, verify=False)
        # res.raise_for_status()
        # res.encoding = 'utf-8'
        # res.encoding = 'utf-8'
        # res.encoding = 'gbk'
        # print(res.text)
        shortcontent_soup = BeautifulSoup(res.text, "lxml")  # from_encoding="utf-8", unicode-escape, .decode('utf-8'), .encode('unicode-escape')
        # shortcontent_soup = BeautifulSoup(res.content.decode('utf-8', 'ignore'), "html.parser")
        # shortcontent_soup = BeautifulSoup(res.text.encode('utf-8', 'ignore'), 'html.parser')
        # print(shortcontent_soup)
        time.sleep(random.uniform(0, 1))

        # dateAndTime = datetime.datetime.fromisoformat(shortcontent_soup.find('span', attrs={'class': 'date'}).get_text().strip())
        # author = shortcontent_soup.find('div', attrs={'class': 'article-info'}).find('span').get_text().strip().replace('\n', '')
        # content = shortcontent_soup.find('div', attrs={'class': 'short-content'}).get_text().strip().replace('\n', '')
        # interactions = [x for x in shortcontent_soup.find_all('p') if '<p><span>' in str(x)][-1].find('span').get_text().strip().replace('\n', '')
        #
        # shortcontent_dict['datetime'] = dateAndTime
        # shortcontent_dict['author'] = author
        # shortcontent_dict['content'] = content
        # shortcontent_dict['interactions'] = interactions

        #   EXTRACTING JSON (IN TERM OF DICTIONARY: DATA) FROM SOUP (SHORTCONTENT_SOUP)
        try:
            script_text = shortcontent_soup.find("script", {"id": "__NEXT_DATA__"}).get_text()  # , type="application/ld+json"
            # script_text = shortcontent_soup.find("script", {"id": "__NEXT_DATA__"}).text.strip()    # , type="application/ld+json"
            data = json.loads(script_text)['props']['pageProps']['shortContentData']  # Dictionary

            contentID = data['contentId']
            contentURL = url
            # title = data['title'].strip()
            content = data['content'].strip()
            # snapcontent = data['snapcontent'].strip()
            ups = data['ups']
            downs = data['downs']
            interactions_bal = data['balance']
            cmts = data['cmts']
            authorID = data['userId']
            authorName = data['userName'].strip()
            authorContentCount = data['authorContentCount']
            authorUpCount = data['authorUpCount']
            money = data['money']
            # createdTime = datetime.datetime.fromtimestamp(data['createTime'] / 1e3) + relativedelta(hours=8)
            createdTime = datetime.datetime.fromtimestamp(data['createTime'] / 1e3)
            importedTime = now

            shortcontent_dict['contentID'] = contentID
            shortcontent_dict['contentURL'] = contentURL
            # shortcontent_dict['title'] = title
            shortcontent_dict['content'] = content
            # shortcontent_dict['snapcontent'] = snapcontent
            shortcontent_dict['ups'] = ups
            shortcontent_dict['downs'] = downs
            shortcontent_dict['interactions_bal'] = interactions_bal
            shortcontent_dict['cmts'] = cmts
            shortcontent_dict['authorID'] = authorID
            shortcontent_dict['authorName'] = authorName
            shortcontent_dict['authorContentCount'] = authorContentCount
            shortcontent_dict['authorUpCount'] = authorUpCount
            shortcontent_dict['money'] = money
            shortcontent_dict['createdTime'] = createdTime
            shortcontent_dict['importTime'] = importedTime

            #   Add on  #
            shortcontent_dict['createdYearMonthDay'] = str(createdTime.year) + f"{createdTime.month:02}" + f"{createdTime.day:02}"
            shortcontent_list.append(shortcontent_dict)

            # pprint.pprint(shortcontent_dict)

        except Exception as e:
            print(traceback.format_exc())
            print("problem found and skipped URL: {}".format(url))
            skip_count += 1
            continue

    df_shortcontent = pd.DataFrame(shortcontent_list)

    if not df_shortcontent.empty:

        #####   FILTER FOR ONLY TODAY   #####
        if FILTER:
            if FILTER_TYPE.upper() == 'D':
                df_shortcontent = df_shortcontent[df_shortcontent['createdYearMonthDay'] == FILTER_DATE_STRING]
            elif FILTER_TYPE.upper() == 'M':
                df_shortcontent = df_shortcontent[df_shortcontent['createdTime'].apply(lambda x: str(x.month) == str(FILTER_MONTH))]
            df_shortcontent.reset_index(drop=True, inplace=True)

        #####   SUMMARY   #####
        title(string="SUMMARY")
        print('skip_count: {}'.format(skip_count))
        print('Latest shortcontent datetime: {}'.format(df_shortcontent['createdTime'].max()))
        print('Earliest shortcontent datetime: {}'.format(df_shortcontent['createdTime'].min()))
        print('\n-----   SHORT CONTENT DATE CHECK    -----')
        df_shortcontent_date_cnt = df_shortcontent['createdYearMonthDay'].value_counts().rename_axis('DATE').reset_index(name='ARTICLE COUNT')
        df_shortcontent_date_cnt.sort_values(by=['DATE'], ascending=False, inplace=True)
        df_shortcontent_date_cnt.reset_index(drop=True, inplace=True)
        print(df_shortcontent_date_cnt)

        #####   JIEBA   #####
        # jieba.enable_paddle()
        # # jieba.load_userdict('txt/keywords.txt')
        # jieba.load_userdict(PROGRAM_ABSOLUTE_PATH + 'txt/keywords.txt')
        # # jieba.analyse.set_stop_words('txt/stopwords.txt')
        # jieba.analyse.set_stop_words(PROGRAM_ABSOLUTE_PATH + 'txt/stopwords.txt')

        df_shortcontent['content_mod'] = df_shortcontent['content'].copy()
        df_shortcontent['content_mod'].fillna('空白', inplace=True)
        # df_shortcontent['content_mod'] = df_shortcontent['content_mod'].apply(lambda x: re.sub("[0-9a-zA-Z]", " ", x))  # Remove digit and letter and keep only Chinese
        df_shortcontent['content_mod'] = df_shortcontent['content_mod'].apply(lambda x: ' '.join(jieba.lcut(x)))
        df_shortcontent['KEYWORDS_TF_IDF'] = df_shortcontent['content_mod'].apply(lambda x: '/'.join(jieba.analyse.extract_tags(x, topK=TOP_KEYWORDS_NUM_PER_ARTICLE, withWeight=False)))
        # df_shortcontent['KEYWORDS_TEXT_RANK'] = df_shortcontent['content_mod'].apply(lambda x: '/'.join(jieba.analyse.textrank(x, topK=TOP_KEYWORDS_NUM_PER_ARTICLE, withWeight=False)))
        # print(df_shortcontent.head().to_string())
        # print(df_shortcontent.shape)

        print('\n-----   JIEBA (KEYWORDS_TF_IDF)    -----')
        # keywords_tf_idf_top_list = [item for sublist in [x.split('/') for x in df_shortcontent['KEYWORDS_TF_IDF']] for item in sublist]
        keywords_tf_idf_top_list = list(np.concatenate([x.split('/') for x in df_shortcontent['KEYWORDS_TF_IDF']]))
        df_shortcontent_keyword_cnt = top_k(iterable=keywords_tf_idf_top_list, k=TOP_KEYWORDS_NUM_SUMMARY)

        # print('\n-----   JIEBA (KEYWORDS_TEXT_RANK)    -----')
        # keywords_text_rank_top_list = list(np.concatenate([x.split('/') for x in df_shortcontent['KEYWORDS_TEXT_RANK']]))
        # top_k(iterable=keywords_text_rank_top_list, k=TOP_KEYWORDS_NUM_SUMMARY)

        #####   EXPORT    #####
        title(string="EXPORT")
        create_directory(CSV_TODAY_PATH)

        print('-----   df_shortcontent samples    -----')
        print(df_shortcontent.head(10).to_string())

        file_detail = 'shortcontent_detail_{0}_{1}.csv'.format(FILTER_DATE_STRING if FILTER else 'all', TODAY_TIME_STRING)
        print('Exporting file ({0}) to csv ({1}) ...'.format(file_detail, CSV_TODAY_PATH))
        df_shortcontent.to_csv(path_or_buf=CSV_TODAY_PATH + file_detail, encoding='utf-8-sig', index=False)

        file_date_cnt = 'shortcontent_date_cnt_{0}_{1}.csv'.format(FILTER_DATE_STRING if FILTER else 'all', TODAY_TIME_STRING)
        print('Exporting file ({0}) to csv ({1}) ...'.format(file_date_cnt, CSV_TODAY_PATH))
        df_shortcontent_date_cnt.to_csv(path_or_buf=CSV_TODAY_PATH + file_date_cnt, encoding='utf-8-sig', index=False)

        file_keyword_cnt = 'shortcontent_keyword_cnt_{0}_{1}.csv'.format(FILTER_DATE_STRING if FILTER else 'all', TODAY_TIME_STRING)
        print('Exporting file ({0}) to csv ({1}) ...'.format(file_keyword_cnt, CSV_TODAY_PATH))
        df_shortcontent_keyword_cnt.to_csv(path_or_buf=CSV_TODAY_PATH + file_keyword_cnt, encoding='utf-8-sig', index=False)

        del df_shortcontent
        del df_shortcontent_date_cnt
        del df_shortcontent_keyword_cnt
        gc.collect()

    else:
        print('DataFrame (df_shortcontent) is empty!')

else:
    print("No article is found.")
    exit()

get_time_elapsed(startTime=startTime)
