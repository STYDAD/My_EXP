import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import re
import random
import math
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import os
import pymongo
from urllib import parse
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
from pymongo.errors import AutoReconnect, OperationFailure
import json
import logging
from bson import ObjectId
import ast
import regex
import difflib
import traceback


class Writ:
    def __init__(self, size=10000, year = 2016):
        
        # 表头配置
        self.Writ_Col = ['数据编号','标题','案号','文书类型','判决日期',
                        '审理法院','检察机关', '相关人员',
                        '审理经过','诉称','法院查明',
                        '本院认为','判决结果','落款']

        # 词袋配置
        self.word_bag = {
                    '文书类型': ['民事裁定', '民事判决', '执行裁定', '刑事判决', '结案通知',
                             '行政裁定', '刑事裁定', '执行通知', '受理案件通知', '执行决定',
                             '支付令', '行政判决', '民事调解', '应诉通知', '其他',

                             '民事裁定书', '民事判决书', '执行裁定书', '刑事判决书', '结案通知书',
                             '行政裁定书', '刑事裁定书', '执行通知书', '受理案件通知书', '执行决定书',
                             '支付令', '行政判决书', '民事调解书', '应诉通知书', '其他'],

                    '原告': ['诉称，', '诉称：', '提出诉讼请求'],
                    '被告': ['辩称，', '辩称：', '未作答辩'],
                    '法院': ['经审查', '经审理查明']
                }
        
        # 词表映射
        self.word_mapping = {
            '民事裁定': '民事裁定书','民事判决': '民事判决书','执行裁定': '执行裁定书',
            '刑事判决': '刑事判决书','结案通知': '结案通知书','行政裁定': '行政裁定书',
            '刑事裁定': '刑事裁定书','执行通知': '执行通知书','受理案件通知': '受理案件通知书',
            '执行决定': '执行决定书','支付令': '支付令','行政判决': '行政判决书',
            '民事调解': '民事调解书','应诉通知': '应诉通知书','其他': '其他'
        }
        
        # 两种匹配方式,两次筛选
        self.pattern_1 = re.compile('<divid=\'2\'[^>]*>(.*?)<\/div>\s*<divid=')
        self.pattern_2 = re.compile('(?:^|>)(?:(?!代理|辩护)[^<\s])+(?:<|$)')
        self.pattern_3 = re.compile('right.*?</div>(.*?)right', flags=re.DOTALL)
        # 备用的文本清洗规则
        self.replace_rules = {'Ｘ':'x', 'ｘ':'x', '×':'&times;', '·':'&middot;', 
                         " ":'', '\'':'', '[':'',']':'', '、':','}
        self.fields = {'_id': 1,'s1': 1,'s7': 1,'s9': 1,'s31': 1,'s22': 1,'s17': 1,
                       's23': 1,'s25': 1,'s26': 1,'s27': 1,'s28': 1,'qwContent': 1}
        
        # 服务器配置
        self.sever_ip = '192.168.11.248:27017/admin'
        self.user_name = parse.quote_plus("root")
        self.password = parse.quote_plus("123QAZwsx")
        self.Client = self.Login(self.sever_ip, self.user_name, self.password)
        self.last_processed_collection, self.last_processed_position, self.Progress = None,None,None
        self.Cursor = None
        self.IfChange = 0; self.IfEnd = 0
        
        # 文档配置
        self.size = size
        self.year = year
        self.Original_Data, self.Length = None,None
        self.Datatable = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        
    # 获取新一轮数据
    def RELOAD(self):
        self.last_processed_collection, self.last_processed_position, self.Progress = self.load_progress()

    # 登录
    def Login(self, IP, USN, PSW):
        Client = pymongo.MongoClient("mongodb://{0}:{1}@{2}".format(USN, PSW, IP))
        Client.server_info()
        if Client.server_info():
            print('数据库连接完成')
            return Client
        else:
            print("无法连接到MongoDB服务器")



    # 初始化游标
    def Initialize_Cursor(self):
        db = self.Client['cpws']
        all_collections = sorted(db.list_collection_names())
        if not self.last_processed_collection:
            self.last_processed_collection = all_collections[0]
        if self.last_processed_position == 0:
            self.Cursor = db[self.last_processed_collection].find({},self.fields).sort("_id", pymongo.ASCENDING).limit(self.size)
        else:
            self.Cursor = db[self.last_processed_collection].find({"_id": {"$gt": ObjectId(self.last_processed_position)}},self.fields).sort("_id", pymongo.ASCENDING).limit(self.size)

# 关闭数据库连接
        
    # 定位游标
    def Locate_Cursor(self):
        # Ensure the cursor is not yet executed
        self.Cursor = self.Client['cpws'][self.last_processed_collection]\
            .find({"_id": {"$gt": self.last_processed_position}},self.fields).sort("_id", pymongo.ASCENDING).limit(self.size)
     
    # 切换DB
    def Skip_DB(self):
        '''        
        db = self.Client['cpws']
        all_collections = sorted(db.list_collection_names())
        for collection_name in all_collections:
            if collection_name <= self.last_processed_collection:
                continue
            else:
                self.IfChange = 0
                self.last_processed_collection = collection_name
                self.last_processed_position = 0
                break'''
        pass

    # 服务器文件读取方式    
    def New_Data(self):
        # 切换collection
        if self.IfChange:
            self.Skip_DB()
        # 切换到末尾
        if self.IfChange:
            self.IfEnd = 1
            return
        all_data = []
        processed_count = 0
        for writ in self.Cursor:
            self.Progress += 1
            processed_count += 1
            processed_data = self.process_data_locally(writ)
            all_data.append(processed_data)
            self.last_processed_position = writ["_id"]
        
        if processed_count < self.size:
            self.IfChange = 1
        Data = pd.DataFrame(all_data)
        null_count = Data['qwContent'].isnull().sum()
        removed_ids = Data.loc[Data['qwContent'].isna(), '_id']
        self.Save_Missing(removed_ids)
        self.Pure_Data = Data.dropna(subset=['qwContent']).reset_index(drop=True)   
        self.Locate_Cursor()
        return self.Pure_Data, self.size - null_count
    
    # 获取新数据
    def Get_Data(self):
        self.Original_Data, self.Length = self.New_Data()
        # print(str(self.Original_Data['_id']))
        self.Datatable = self.New_Table(self.Length)
        
    
    def Save_Missing(self, series):
        filename = f'Loss_{self.year}.txt'
        mode = 'a' if os.path.exists(filename) else 'w'
        with open(filename, mode) as file:
            if isinstance(series, pd.Series):
                for value in series.values:
                    file.write(str(value) + '\n')
            else:
                file.write(f"{series}\n")


    def process_data_locally(self,batch):
        selected_columns = ['_id','s1', 's7', 's9', 's31', 's22', 's17', 's23', 's25', 's26', 's27', 's28', 'qwContent']
        processed_data = {key: batch.get(key, None) for key in selected_columns}
        processed_data['_id'] = str(processed_data['_id'])
        return processed_data
    
    
    # 进度保存
    def save_progress(self, collection_name, current_position, current_progress):
        filename = 'progress_' + self.year + '.txt'
        with open(filename, 'w') as file:
            file.write(f"{collection_name},{current_position},{current_progress}")
    
    # 进度读取
    def load_progress(self):
        try:
            filename = 'progress_' + self.year + '.txt'
            with open(filename, 'r') as file:
                content = file.read().strip().split(',')
                collection_name = content[0]
                current_position = content[1]
                current_Progress = int(content[2])
            return collection_name, current_position, current_Progress
        except FileNotFoundError:
            return 'ws_'+ self.year, 0, 0
    
    
    
    # 本地文件读取方式
    ''' 
    def Get_Data(self):
        Data = pd.read_csv('Origin.csv', usecols=['s1', 's7', 's9', 's31', 's22', 's17', 's23', 's25', 's26', 's27', 's28', 'qwContent'])
        null_count = Data['qwContent'].isnull().sum()
        if null_count == 0:
            Pure_Data = Data.copy()
        else:
            Data_cleaned = Data.dropna(subset=['qwContent']).reset_index(drop=True)
            Pure_Data = Data_cleaned
        return Pure_Data, self.size - null_count
    '''

    
    def New_Table(self,Len):
        self.cleaned_html = pd.DataFrame(index=range(Len), columns = ['text'])
        return pd.DataFrame(index=range(Len), columns = self.Writ_Col)
    
    def Refresh(self):
        self.Datatable = None
        self.Original_Data = None
    
    def remove_newlines_without_period(self,text):
    # Use a regular expression to replace '\n' not preceded by '。'
        return re.sub(r'(?<!。)\n', '', text)


    # 表格克隆
    def Clone_table(self):
        self.Datatable['数据编号'] = np.where(pd.isnull(self.Original_Data['_id']), '无', self.Original_Data['_id'])
        self.Datatable['标题'] = np.where(pd.isnull(self.Original_Data['s1']), '无', self.Original_Data['s1'])
        self.Datatable['案号'] = np.where(pd.isnull(self.Original_Data['s7']), '无', self.Original_Data['s7'])
        self.Datatable['判决日期'] = np.where(pd.isnull(self.Original_Data['s31']), '无', self.Original_Data['s31'])
        # self.Datatable['审理经过'] = np.where(pd.isnull(self.Original_Data['s23']), '无', self.Original_Data['s23'])
        vectorized_func = np.vectorize(self.remove_newlines_without_period)
        self.Datatable['本院认为'] = vectorized_func(np.where(pd.isnull(self.Original_Data['s26']), '无', self.Original_Data['s26']))
        self.Datatable['判决结果'] = vectorized_func(np.where(pd.isnull(self.Original_Data['s27']), '无', self.Original_Data['s27']))
        self.Datatable['落款'] = np.where(pd.isnull(self.Original_Data['s28']), '无', self.Original_Data['s28'])
    
    # 头部数据清洗
    def Head_Data_Cleaning(self):
        # Set保存处理好的头部数据：审理法院、文书类型、检察机关
        s22_data = self.Original_Data['s22'].replace(r'\s+', '').str.split('\n', expand=True)

        def clean_row(row):
            Set = ['其他', '其他', '无']
            for element in row:
                if not pd.notnull(element):
                    break
                if len(element) > 20:
                    element = element[-20:]
                if '检察' in str(element):
                    Set[2] = element
                Is_Court = '法院' in str(element)
                matched_words = [word for word in self.word_bag['文书类型'] if word in str(element)]

                if Is_Court and not matched_words:
                    Set[0] = element
                elif Is_Court and matched_words:
                    court_index = str(element).index('法院')
                    keyword_index = min(
                        str(element).index('书') if '书' in str(element) else len(str(element)),
                        str(element).index('令') if '令' in str(element) else len(str(element))
                    )
                    Set[0] = str(element)[:court_index]
                    Set[1] = self.word_mapping.get(matched_words[-1], matched_words[-1])
                elif not Is_Court and matched_words:
                    Set[1] = self.word_mapping.get(matched_words[-1], matched_words[-1])
            return Set[0], Set[1], Set[2]


        futures = [self.executor.submit(clean_row, row) for index, row in s22_data.iterrows()]
        results = [future.result() for future in futures]
        self.Datatable[['审理法院', '文书类型', '检察机关']] = pd.DataFrame(results, index=s22_data.index)
        # results = s22_data.apply(clean_row, axis=1)

        
        
    def fuzzy_match(self, pattern, text):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None

    # 模糊匹配算法
    def Fuzy_Match(self, name, text):
        Match_Text = '其他'
        if name:
            Name = name
            fuzzy_name_pattern = re.compile(f"([^。]*?{re.escape(Name)}[^。]*。)")
            matched_text = self.fuzzy_match(fuzzy_name_pattern, text)
            # print(Name,matched_text)
            if matched_text:
                similarity_ratio = fuzz.ratio(Name, matched_text)
                
                if similarity_ratio > 0:  # 存在相似值
                    # self.cleaned_html = re.sub(matched_text, '', self.cleaned_html)
                    Match_Text = matched_text
        
        return Match_Text
    
    # 相关人员匹配
    def Plaintiff_Defendant(self):
        futures = [self.executor.submit(self.process_sample, row) for index, row in self.Original_Data.iterrows()]
        results = [future.result() for future in futures]
        self.Datatable['相关人员'] = results

    def process_sample(self, row):
        Set = ['其他']
        current_row = row.name
        try:
            Names = row['s17'] if type(row['s17']) == list else ast.literal_eval(row['s17'])
        except (ValueError, TypeError):
            Names = ['None']  

        if '通知书' in self.Datatable['文书类型'].loc[current_row]:
            Set = Names
        else:
            self.cleaned_html.iloc[current_row] = self.clean_html(row['qwContent'])
            FuzzySet = []
            for Name in Names:
                if self.Fuzy_Match(Name, self.cleaned_html.iloc[current_row][0]) == '其他':
                    break
                else:
                    FuzzySet.append(self.Fuzy_Match(Name, self.cleaned_html.iloc[current_row][0]))
                Set = FuzzySet
        return Set
    
        # 姓名抽取函数
    def extract_names(self, name_list):
        try:
            names = re.sub('|'.join(re.escape(k) for k in self.replace_rules.keys()), \
                           lambda m: self.replace_rules[m.group(0)], name_list).split(',')
            return names
        except Exception as e:
            names = ['None']
            return names

    def clean_html(self, html):
        cleaned_html = re.sub(r'\s|&nbsp;', '', html, flags=re.MULTILINE | re.DOTALL).lower()
        cleaned_html = re.sub(f'，，', '，', cleaned_html)
        match = re.search(self.pattern_3, cleaned_html)
        if match:
            extracted_text = match.group(1)
            cleaned_html = re.sub(r'<[^>]*>', '', extracted_text)
            cleaned_html = re.sub(r'<[^>]*$', '', cleaned_html)
        cleaned_html = re.sub(f'，，', '，', cleaned_html, flags=re.MULTILINE | re.DOTALL)
        return cleaned_html


    #  文本处理函数
    def Main_Text_Cleaning(self):
        futures = [self.executor.submit(self.process_text, text) for text in self.Original_Data['s25']]
        results = [future.result() for future in futures]
        self.Datatable[['诉称', '法院查明']] = results

    def process_text(self, text):
        if pd.isnull(text):
            Set = ['无','无']
        else:
            Set = ['','']
            s25_data = text.replace(r'\s+', '').split('\n')
            for Word in s25_data:
                if not pd.notnull(Word):
                    break
                Sentence = re.split(r'(?<=,)', Word)[0]
                # Is_Plaintiff = [word for word in self.word_bag['原告'] if word in str(Sentence)]
                # Is_Defendant = [word for word in self.word_bag['被告'] if word in str(Sentence)]
                Is_court = [word for word in self.word_bag['法院'] if word in str(Sentence)]
                if Is_court:
                    Set[1] += Word + '。\n'
                else:
                    Set[0] += Word + '\n'

            for t in range(2):
                if Set[t] == '' or len(Set[t]) < 5:
                    Set[t] = '无'

        return pd.Series({'诉称': Set[0], '法院查明': Set[1]})
    

    # 经历处理
    def Process_Pass(self):
        # HTML = [HTML_1, HTML_2 ... HTML_n]
        HTML = self.cleaned_html.squeeze().apply(lambda x: x.replace('\n', '').split('。') if pd.notnull(x) else '无')
        # Process_pass = [Process_1[0],Process_2[0] ... Process_n[0]]
        Process_pass = self.Pure_Data['s23'].apply(lambda x: x.replace('\n', '').split('。') if x else x)

        # 遍历每一个处理好的HTML
        for index_1, row in enumerate(HTML):
            # 如果没有经历，设为无，并处理下一份数据
            if not Process_pass[index_1]:
                self.Datatable['审理经过'].iloc[index_1] = '无'
                continue

            # 如果有经历，则截取HTML的结尾
            sent = Process_pass[index_1][-1]
            for index_2, sent_2 in enumerate(row):
                similarity = fuzz.token_sort_ratio(sent, sent_2)
                if similarity > 80:
                    # 更新DataFrame中的对应行
                    row = row[:index_2 + 1]
                    break
            
            # 取第i个经历的第一句话
            pp = Process_pass[index_1][0]
            for index_2, sent_2 in enumerate(row):
                similarity = fuzz.token_sort_ratio(pp, sent_2)
                if similarity > 80:
                    # 更新DataFrame中的对应行
                    row = row[index_2:]
                    break

            # 更新Datatable中的对应行
            # 假设HTML的每一行是一个字符串列表，我们将它们连接成一个字符串
            self.Datatable['审理经过'].iloc[index_1] = '。'.join(row) + '。'
            


    # 将Datatable保存为jsonl格式的文件
    def save_to_jsonl(self):
        save_path = './'
        filename = str(self.last_processed_collection) + str(int(self.Progress/(self.size*100))).zfill(5) + '.jsonl'
        save_path = save_path + str(self.last_processed_collection) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, filename), 'a', encoding='utf-8') as file:
            for _, row in self.Datatable.iterrows():
                json_data = row.to_dict()
                json.dump(json_data, file, ensure_ascii=False)
                file.write('\n')
        file.close()
        self.save_progress(self.last_processed_collection, self.last_processed_position, self.Progress)
    
    def save_error_index(self):
        file_name = f'Error_{self.last_processed_collection}.txt'
        try:
            with open(file_name, 'a') as file:
                file.write(f"{self.last_processed_position}\n")  # 添加换行符，使每个错误索引占一行
        except Exception as e:
            print(f"Error while saving error index: {e}\n")
        finally:
            if file and not file.closed:
                file.close()

    def ReLogin(self):
        self.Client = self.Login(self.sever_ip, self.user_name, self.password)
        self.Initialize_Cursor()
        self.Client.server_info()

class ReconnectionError(Exception):
    pass

def reconnect_with_retry(reconnect_function, max_retries, delay_seconds):
    
    for attempt in range(1, max_retries + 99999999999):
        try:
            reconnect_function()
            print(f"Reconnection successful on attempt {attempt}\n")
            logging.info(f"Reconnection successful on attempt {attempt}\n")
            return  # 如果成功则退出函数
        except Exception as e:
            print(f"Reconnection attempt {attempt} failed. Error: {e}\n")
            logging.info(f"Reconnection attempt {attempt} failed. Error: {e}\n")
            time.sleep(delay_seconds)
    
    # 如果达到最大重试次数仍然失败，则抛出自定义异常
    logging.info(f"Maximum retry attempts reached. Unable to reconnect\n")
    raise ReconnectionError("Maximum retry attempts reached. Unable to reconnect.")