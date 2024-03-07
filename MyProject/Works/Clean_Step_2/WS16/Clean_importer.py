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
import threading
lock = threading.Lock()

class Writ:
    def __init__(self, size=10000, year = '2016', filename = ''):
        
        # 表头配置
        self.Writ_Col = ['数据编号','标题','案号','相关人员','审理经过','诉称','法院查明','本院认为','诉称与法院查明','s23','案件类型']

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
                    '法院': ['经审查', '经审理查明'],
                    '诉称': ['诉称，', '诉称：', '提出诉讼请求', '辩称，', '辩称：', '未作答辩', '上诉请求','审查认为'],
                    '头部': ['本院','审判员','诉称','保全申请']

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
        # self.pattern_3 = re.compile('right.*?</div>(.*?)right', flags=re.DOTALL)
        self.pattern_3 = re.compile('right.*?</div>(.*)', flags=re.DOTALL)
        self.pattern_4 = re.compile('<.*?right>.*?<.*?>', flags=re.DOTALL)
        # 备用的文本清洗规则
        self.replace_rules = {'Ｘ':'x', 'ｘ':'x', '×':'&times;', '·':'&middot;', 
                         " ":'', '\'':'', '[':'',']':'', '、':','}
        self.fields = {'_id': 1,'s1': 1,'s7': 1,'s8': 1,'s9': 1,'s31': 1,'s22': 1,'s17': 1,
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
        self.filename = filename

        # 修改Json配置
        self.last_processed_json = self.filename
        self.last_json_position = None
        self.json_progress = None
        self.json_path = 'ws_' + self.year
        

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
    
    def remove_newlines_without_period(self,text):
    # Use a regular expression to replace '\n' not preceded by '。'
        return re.sub(r'(?<!。)\n', '', text)
    
    def Process_Text(self):

        def replace_tag_1(match):
            tag = match.group(0)
            id_match = re.search(r"id", tag)
            if id_match:
                return f'({id_match.group(0)})'
            else:
                return ''

        def replace_tag_2(match):
            preceding_char = match.string[match.start()-1] if match.start() != 0 else ''
            if preceding_char == '。':
                return '(id)'
            else:
                return ''

        def Is_Formulated(text):
            if text == "":
                return False
            lines = text.split('\n')
            for line in lines[:-1]:
                if not line.endswith('。'):
                    return False
            return True

        Index = 0
        for row in self.Datatable.itertuples():
            flag = 0
            HTML = self.Original_Data['qwContent'].iloc[Index]
            cleaned_html = re.sub(r'\s|&nbsp;|', '', HTML, flags=re.MULTILINE | re.DOTALL).lower()
            
            cleaned_html = re.sub(f'，，', '，', cleaned_html, flags=re.MULTILINE | re.DOTALL)
            # cleaned_html = re.sub('××|×|&times;', '某', cleaned_html)
            match = re.search(self.pattern_3, cleaned_html)
            if match:
                extracted_text = match.group(1)
                cleaned_html = re.sub(r'<.*?right>.*?<.*?>', '', extracted_text)
                cleaned_html = re.sub(r'<[^>]*>', replace_tag_1, extracted_text)
                cleaned_html = re.sub(r"\(id\)", replace_tag_2, cleaned_html)
                cleaned_html = re.sub(r'<.*', '', cleaned_html)
            if self.Datatable['本院认为'].iloc[Index] == '':
                self.Datatable['本院认为'].iloc[Index] = '无'
            if self.Datatable['本院认为'].iloc[Index] != '无': 
                Sentence_to_Del = self.Datatable['本院认为'].iloc[Index].split('。')[0]
                index_to_del = cleaned_html.find(Sentence_to_Del)
                if index_to_del != -1:
                    cleaned_html = cleaned_html[:index_to_del]
            index = cleaned_html.find('审理终结。')
            if index != -1 and not cleaned_html[index:].startswith('审理终结。(id)'):
                cleaned_html = cleaned_html[:index] + '审理终结。(id)' + cleaned_html[index+len('审理终结。'):]
            paragraphs = re.split(r"\(id\)", cleaned_html)

            
            # 检查
            if not pd.isnull(self.Original_Data['s23'].iloc[Index]) and Is_Formulated(self.Original_Data['s23'].iloc[Index]):
                text = self.Original_Data['s23'].iloc[Index]
                sentences = text.split('。')
                first_sentence = sentences[0] if len(sentences) > 0 else ''
                last_sentence = sentences[-1] if len(sentences) > 1 else first_sentence

                if_Find = paragraphs[0].find(first_sentence)

                cleaned_html = re.sub(r"\(id\)", '', cleaned_html)
                first_index = cleaned_html.find(first_sentence + '。')
                last_index = cleaned_html.find(last_sentence + '。')

                if if_Find:
                    if first_index != -1:
                        paragraphs[0] = cleaned_html[:first_index]
                    if last_index != -1:
                        if len(paragraphs) >= 2:
                            paragraphs[1] = cleaned_html[first_index:last_index + len(last_sentence + '。')] 
                        if len(paragraphs) >= 3:
                            paragraphs[2] = cleaned_html[last_index + len(last_sentence + '。'):]

            
            for i,para in enumerate(paragraphs):
                if para:
                    if para[-1] == '，':
                        paragraphs[i] = para[:-1] + '。'

            List = ['相关人员','审理经过','诉称与法院查明']
            for i in range(3):
                if i >= len(paragraphs):
                    self.Datatable[List[i]].iloc[Index] = '无'
                else:
                    self.Datatable[List[i]].iloc[Index] = paragraphs[i]
            self.Datatable['案件类型'].iloc[Index] = self.Original_Data['s8'].iloc[Index]
            Index += 1
        
        sc_list = []
        cm_list = []

        for i in self.Datatable['诉称与法院查明'].values:
            data = self.process_sc_cc(i)
            sc_list.append(data[0])
            cm_list.append(data[1])
        self.Datatable['诉称'] = sc_list
        self.Datatable['法院查明'] = cm_list
        self.Datatable = self.Datatable.drop('诉称与法院查明', axis=1)

    def check_word_in_sen(self, sen, words, pure):
        index = None
        for word in words:
            if word in sen:
                index = pure.index(sen)

            else:
                continue
        return index

    def process_sc_cc(self,text):
        if pd.isnull(text):
            Set = ['无', '无']
            return pd.Series({'诉称': Set[0], '法院查明': Set[1]})
        else:
            Set = ['', '']
            pure_data = text.replace('\n', '')
            s25_data = [i.replace('\n', '') for i in text.replace(r'\s+', '').split('。')]
            # 先找到法院的开始句子
            court_index = None
            futures = [self.executor.submit(self.check_word_in_sen, i, self.word_bag['法院'], s25_data) for i in s25_data]
            results = [i.result() for i in futures]
            for result in results:
                if result is not None:
                    court_index = result
                    break
            if court_index is not None:
                # 有的话直接分割，对应加入到诉称和法院里面
                tar_sen = s25_data[court_index]
                Set[0] = pure_data[:pure_data.find(tar_sen)]
                Set[1] = pure_data[pure_data.find(tar_sen):]
            else:
                # 没有的话先判断有没有原告，被告的词袋，没有的话就加入到法院，有的话加入到诉称
                talk_logo = 0
                futures = [self.executor.submit(self.check_word_in_sen, i, self.word_bag['诉称'], s25_data) for i in s25_data]
                results = [i.result() for i in futures]
                for i in results:
                    if i is not None:
                        talk_logo = 1
                        break
                if talk_logo == 0:
                    Set[0] = '无'
                    Set[1] = pure_data
                else:
                    Set[0] = pure_data
                    Set[1] = '无'

            if Set[0] is None or Set[0] == '，' or Set[0] == '':
                Set[0] = '无'
            if Set[1] is None or Set[1] == '，' or Set[1] == '':
                Set[1] = '无'
            return pd.Series({'诉称': Set[0], '法院': Set[1]})

    def load_Json_Progress(self):
        Json_Save = self.year + 'Plan.txt'
        with open(Json_Save, 'r') as file:  
            for line in file:
                lines = line.split(',')
                first_part = lines[0]
                if self.filename == first_part:
                    Json_position = lines[1]
                    current_Progress = int(lines[2])
        return Json_position, current_Progress

    def Get_Json(self):
        Absolute_Json = self.json_path + '/' + self.last_processed_json
        with open(Absolute_Json, 'r', encoding='utf-8') as file:
            
            lines = file.readlines()
            try:
                lines_to_modify = lines[self.last_json_position:self.last_json_position + self.size]
                self.last_json_position = int(self.last_json_position) + self.size
            except Exception as e:
                print(e)
                lines_to_modify = lines[self.last_json_position:]
                

            
            data_dicts = []
            for line in lines_to_modify:
                data_dict = json.loads(line)
                data_dicts.append(data_dict)
            df = pd.DataFrame(data_dicts)

            print(df)
        return df    
    
    def Save_Json_progress(self):
        
        filename = self.year + 'Plan.txt'
        self.json_progress = self.json_progress + self.Datatable.shape[0]
        
        with open(filename, 'r') as file:
            lines = file.readlines()
        for i in range(len(lines)):
            first_part = lines[i].split(',')[0]
            if first_part == self.filename:
                lines[i] = f"{self.filename},{self.last_json_position},{self.json_progress}\n"
                break

        with lock:
            with open(filename, 'w') as file:
                file.writelines(lines)

    def Save_Json_Modified(self):
        save_path = './'
        filename = self.last_processed_json
        save_path = save_path + 'WS_Jsonl_' + self.year + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with lock:
            with open(os.path.join(save_path, filename), 'a', encoding='utf-8') as file:
                for _, row in self.Datatable.iterrows():
                    json_data = row.to_dict()
                    json.dump(json_data, file, ensure_ascii=False)
                    file.write('\n')
            file.close()

        if self.Datatable.shape[0] != self.size:
            return False

        self.Save_Json_progress()
        return True

    def Search_ID(self, ids_List):
        object_ids_list = [ObjectId(id) for id in ids_List if ObjectId.is_valid(id)]
        collection = self.Client['cpws']['ws_' + self.year]
        query = {'_id': {'$in': object_ids_list}}
        projection = {'_id': 1, 'qwContent': 1, 's8': 1, 's23': 1}
        results = collection.find(query, projection)
        df = pd.DataFrame(list(results))
        return df

    def Reload_JsonL_Index(self):
        self.last_json_position, self.json_progress = self.load_Json_Progress()
        self.last_json_position = int(self.last_json_position)

    def Modify_Jsonl(self):
        self.Datatable = self.Get_Json()
        self.Datatable['案件类型'] = 'None'
        self.Datatable['诉称与法院查明'] = ''
        ID_List = self.Datatable['数据编号'].tolist()
        self.Original_Data = self.Search_ID(ID_List)
        self.Process_Text()


































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
        file_name = f'Error_{self.year}.txt'
        try:
            with open(file_name, 'a') as file:
                file.write(f"{self.last_json_position}\n")  # 添加换行符，使每个错误索引占一行
        except Exception as e:
            print(f"Error while saving error index: {e}\n")
        finally:
            if file and not file.closed:
                file.close()

    def ReLogin(self):
        self.Client = self.Login(self.sever_ip, self.user_name, self.password)
        self.Initialize_Cursor()
        self.Client.server_info()


