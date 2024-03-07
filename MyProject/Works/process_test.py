import os
import threading
import time

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

from prompt.utils import extract_query, extract_answer_query
from pg.models import AbtestQuery, AbtestRef, AbtestRefV2, ABTestBatch, abtest_ref_table, abtest_batch_table
from pg.utils import pg_pool
from es.utils import get_refs,bm25search

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import BGE_RERANK_PATH, BGE_RERANK_FT_PATH
from law_uuid_100_0102 import LAW_UUID_100_DICT
from pdf_uuid_1222_02 import dict_uuid
from tokenizer.jieba_tokenizer import tokenizer, law_titles_list
from tokenizer.legal_terminology import law_explain_dict, law_mapping_dict
from tokenizer.law_titles_dict import law_title_mapping_dict
from es.es_searcher import ESSearcher

import logging
from logger import get_logger

logger = get_logger(__name__, logging.DEBUG)

def supplement_input(input):
    words = tokenizer(input)
    law_title = ''
    for word in words:
        if word in law_mapping_dict.keys():
            input = input.replace(word, law_mapping_dict[word])
        if word in law_title_mapping_dict.keys():
            input = input.replace(word, law_title_mapping_dict[word])
            law_title = law_title_mapping_dict[word]
        if word in law_explain_dict.keys():
            input += law_explain_dict[word]
        if word in law_titles_list:
            law_title = law_title_mapping_dict[word]
    return [input, law_title]


def filter_handle(input, es_filter, law_title):
    filter_list = ['拟上市公司', '拟上市企业', 'IPO', 'ipo', '报告期', '首次公开发行并上市', '首发上市', '首发申报',
                   '申请上市', '上市申请']
    # for el in filter_list:
    #     if el in input:
    #         es_filter.append({"term": {"metadata.priority.keyword": "0"}})
    if law_title:
        es_filter.append({"term": {"metadata.title.keyword": law_title}})
    return es_filter


def question_to_query(input: str, query_kind: int, batch_id: int, index, sub_query):
    ## 根据法律专有名词词典补充描述问题
    input = input.replace("请问根据法律法规的规定", "")
    input = input.replace("请问根据法律法规规定", "")
    input = input.replace("根据法律法规规定", "")
    input = input.replace("请问根据法律法规", "")
    input = input.replace("根据法律法规", "")
    input = input.replace("拟上市公司", "")
    input = input.replace("\n","")
    input = input.replace("/n","")
    if input[0] == ',' or input[0] == '，':
        input = input[1:len(input)]
    logger.debug(f"耗时:{int(time.time()) - mainQueryStart}s")
    results = supplement_input(input)
    logger.debug(f"耗时:{int(time.time()) - mainQueryStart}s")
    query = results[0]
    law_title = results[1]
    logger.debug(f'query:{query}, law_title:{law_title}')
    es_filter = []
    # es_filter = filter_handle(input, es_filter, law_title)
    logger.info(f'query:{query}')
    query_to_ref(input, query, query_kind, batch_id, index, sub_query, law_title, es_filter)
    """
    input表示原始问题
    query表示用于获取ref的输入
    """


"""
根据问题编号和doc_id，返回是否击中 0:否 1:是
"""


def get_scored(index, doc_id):
    result = 0
    if index != 0:
        true_result_id = dict_uuid.get(str(index))
        # print(f'doc_id:{doc_id},true_id:{true_result_id}')
        if doc_id == true_result_id:
            result = 1
    return result


def get_law_hit_scored(index, doc_id):
    result = 0
    if index != 0:
        true_result_id = LAW_UUID_100_DICT.get(str(index))
        # print(f'doc_id:{doc_id},true_id:{true_result_id}')
        if doc_id in true_result_id:
            result = 1
    return result


def insert_table(docs: list, input, index, query, query_kind, batch_id, query_answer):
    law_id_list = [doc.metadata.get('law_id') for doc in docs]
    true_result_list = LAW_UUID_100_DICT.get(str(index))
    total_ture_num = len(true_result_list)
    ture_num = 0
    for law_id in true_result_list:
        if law_id in law_id_list:
            ture_num += 1
    true_score = round(ture_num / total_ture_num, 1)
    query = query_answer if query_answer is not None and query_answer != '' else query
    logger.debug(f'query_answer:{query_answer}, query:{query}')
    for doc in docs:
        try:
            title = doc.metadata.get('title')
            _id = doc.metadata.get('_id')
            law_id = doc.metadata.get('law_id')
            hit_score = get_law_hit_scored(index, law_id)
            score_bm25 = doc.metadata.get('score_bm25')
            rerank_score = doc.metadata.get('rerank_score')
            embedding_score = 0
            if score_bm25 is None:
                embedding_score = doc.metadata.get('_score')
            page_content = doc.page_content.strip()
        except Exception as e:
            logger.error(f"get_refs异常: {e}")
            continue

        dt = datetime.now()
        try:
            ref_model = AbtestRefV2(
                batch_id=batch_id,
                question=input,
                query_text=query,
                query_kind=query_kind,
                query_type=index,
                es_id=_id,
                ref_title=title,
                ref_text=page_content,
                create_time=dt,
                update_time=dt,
                ref_score_gpt1=true_score,
                scored=hit_score,
                ref_score_gpt2=embedding_score,
                ref_score_gpt3=score_bm25,
                ref_rerank_score=rerank_score
            )
        except:
            ref_model = AbtestRefV2(
                batch_id=batch_id,
                question=input,
                query_text=query,
                query_kind=query_kind,
                query_type=index,
                es_id=_id,
                ref_title=title,
                ref_text=page_content,
                create_time=dt,
                update_time=dt,
                ref_score_gpt1=true_score,
                scored=hit_score,
                ref_score_gpt2=embedding_score,
                ref_score_gpt3=score_bm25,
                ref_rerank_score=rerank_score
            )

        pg_pool.execute_insert(abtest_ref_table, ref_model)

def query_to_ref(input: str, query: str, query_kind: int, batch_id: int, index: int, sub_query: list, law_title,
                 es_filter: list = list()):
    logger.debug(f"耗时:{int(time.time()) - mainQueryStart}s")
    es_filter = filter_handle(input, es_filter, law_title)

    bm25_fake_query_list = []
    eb_fake_query_list = []
    g_fake_answer = ''
    def fake_answer_thread_worker(bm25_fake_query_list, eb_fake_query_list):
        logger.debug(f"fake_answer_thread_worker开始耗时:{int(time.time()) - mainQueryStart}s")
        global g_fake_answer, g_ESSearcher_in_child_thread
        g_fake_answer = extract_answer_query(query, query_kind, True, True)
        logger.debug(f"fake_answer_thread_worker 制作伪答案耗时:{int(time.time()) - mainQueryStart}s")
        logger.info(f'g_fake_answer:{g_fake_answer}')
        bm25_fake_query_list.extend(g_ESSearcher_in_child_thread.ESbm25search(g_fake_answer, top_k, query_kind, es_filter))
        logger.debug(f"fake_answer_thread_worker bm25 耗时:{int(time.time()) - mainQueryStart}s")
        eb_fake_query_list.extend(g_ESSearcher_in_child_thread.return_refs(g_fake_answer, query_kind, top_k, es_filter))
        logger.debug(f"fake_answer_thread_worker embedding 耗时:{int(time.time()) - mainQueryStart}s")

    fake_answer_thread = threading.Thread(target=fake_answer_thread_worker, args=(bm25_fake_query_list, eb_fake_query_list));
    fake_answer_thread.start()

    global g_ESSearcher_in_main_thread
    logger.debug(f"耗时:{int(time.time()) - mainQueryStart}s")
    bm25_list = g_ESSearcher_in_main_thread.ESbm25search(query, top_k, query_kind, es_filter)
    logger.debug(f"bm25search耗时:{int(time.time()) - mainQueryStart}s")

    eb_list = g_ESSearcher_in_main_thread.return_refs(query, query_kind, top_k, es_filter)
    logger.debug(f"get_refs耗时:{int(time.time()) - mainQueryStart}s")
    eb_avg_score = sum(i.metadata.get('_score') for i in eb_list) / top_k
    logger.info(f'问题编号：{index}, eb均值为：{eb_avg_score}')

    fake_answer_thread.join()
    docs = bm25_list + eb_list + bm25_fake_query_list + eb_fake_query_list
    insert_table(docs, input, index, query, query_kind, batch_id, g_fake_answer)
    # rerank(bm25_list, eb_list, bm25_fake_query_list, eb_fake_query_list, batch_id, input, index, query, query_kind, g_fake_answer, top_k)
    logger.debug(f"rerank耗时:{int(time.time()) - mainQueryStart}s")


def read_file_lines(file_path):
    inputs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            input = line.strip()
            if input:
                inputs.append(line.strip())
    return inputs


def create_batch(filename, batch_name, user_id):
    """
    获取batchID时未考虑并发写入batch的情况
    """
    dt = datetime.now()
    model = ABTestBatch(
        file_name=filename,
        batch_name=batch_name,
        create_time=dt,
        update_time=dt,
        user_id=user_id
    )
    pg_pool.execute_insert(abtest_batch_table, model)

    sql = 'select max(id) from {}'.format(abtest_batch_table)
    r = pg_pool.execute_query(sql, ())
    return r[0][0]


def rerank(bm25_list: list, eb_list: list, bm25_fake_query_list: list, eb_fake_query_list: list, batch_id, input, index, query,
           query_kind, fake_query, top_k):

    lists_to_check = {"bm25_list": bm25_list, "eb_list": eb_list, "bm25_fake_query_list": bm25_fake_query_list, "eb_fake_query_list": eb_fake_query_list}

    # original bm25_list and eb_list
    doc_lists = []

    for name, lst in lists_to_check.items():
        if lst is None or len(lst) < 1:
            logger.warning(f'{name} 为空')
        else:
            logger.debug(f'Adding {name} length:{len(lst)}')
            doc_lists.append(lst)

    rerank_strategy = "rrf_v2"

    match rerank_strategy:
        case "none": # without any rerank
            insert_table(doc_lists, input, index, query, query_kind, batch_id, fake_query)

        case "bge_rerank":
            all_chunks = []
            for doc_list in doc_lists:
                for doc in doc_list:
                    all_chunks.append(doc)
            logger.debug(f"耗时:{int(time.time()) - mainQueryStart}s")
            sort_list = bge_ranking(input, all_chunks, False, 10)
            logger.debug(f"bge_reranking耗时:{int(time.time()) - mainQueryStart}s")
            insert_table(sort_list, input, index, query, query_kind)
            logger.debug(f"insert_table耗时:{int(time.time()) - mainQueryStart}s")
        case "rrf":
        ######## rrf rerank start#########
            all_chunks = set()
            for doc_list in doc_lists:
                for doc in doc_list:
                    all_chunks.add(doc.page_content)
            weights = [0.25, 0.25, 0.25, 0.25]
            subquery_set_size = len(doc_lists)
            logger.debug(f'subquery_set_size:{subquery_set_size}')
            weights = [round(1/subquery_set_size,2) for i in range(subquery_set_size)]
            rrf_score_dic = {chunk: 0.0 for chunk in all_chunks}
            logger.debug(f"rrf 耗时1:{int(time.time()) - mainQueryStart}s")
            for doc_list, weight in zip(doc_lists, weights):
                for rank, doc in enumerate(doc_list, start=1):
                    rrf_score = weight * (1 / (rank + subquery_set_size * 30))
                    rrf_score_dic[doc.page_content] += rrf_score
            logger.debug(f"rrf 耗时2:{int(time.time()) - mainQueryStart}s")

            # Sort documents by their RRF scores in descending order
            sorted_documents = sorted(
                rrf_score_dic.keys(), key=lambda x: rrf_score_dic[x], reverse=True
            )
            logger.debug(f"rrf sorted:{int(time.time()) - mainQueryStart}s")
            # Map the sorted page_content back to the original document objects
            page_content_to_doc_map = {
                doc.page_content: doc for doc_list in doc_lists for doc in doc_list
            }
            logger.debug(f"rrf page_content_to_doc_map:{int(time.time()) - mainQueryStart}s")
            sorted_docs = [
                page_content_to_doc_map[page_content] for page_content in sorted_documents
            ]
            logger.debug(f"rrf sorted_docs:{int(time.time()) - mainQueryStart}s")
            print("sortlen:" + str(len(sorted_docs)))
            insert_table(sorted_docs[:10], input, index, query, query_kind, batch_id, fake_query)
            logger.debug(f"rrf insert_table:{int(time.time()) - mainQueryStart}s")
            ######## rrf rerank end#########

        #socre = rrf_score * bge_score
        case "rrf_v2":
            logger.debug(f"rrf_v2 耗时:{int(time.time()) - mainQueryStart}s")

            all_chunks = set()
            all_docs = []
            for doc_list in doc_lists:
                for doc in doc_list:
                    all_docs.append(doc)
                    all_chunks.add(doc.page_content)

            #calculate the rerank score for all docs
            calc_bge_score(query, all_docs, False)
            logger.debug(f"rrf_v2 calc_bge_score 耗时:{int(time.time()) - mainQueryStart}s")

            weights = [0.25, 0.25, 0.25, 0.25]
            subquery_set_size = len(doc_lists)
            rrf_score_dic = {chunk: 0.0 for chunk in all_chunks}
            for doc_list, weight in zip(doc_lists, weights):
                for rank, doc in enumerate(doc_list, start=1):
                    rrf_score = weight * (1 / (rank + subquery_set_size * 30))
                    rerank_score = doc.metadata['rerank_score']
                    total_r_score = rrf_score * rerank_score
                    rrf_score_dic[doc.page_content] += total_r_score

            logger.debug(f"rrf_v2 耗时2:{int(time.time()) - mainQueryStart}s")
            # Sort documents by their RRF scores in descending order
            sorted_documents = sorted(
                rrf_score_dic.keys(), key=lambda x: rrf_score_dic[x], reverse=True
            )
            logger.debug(f"rrf sorted:{int(time.time()) - mainQueryStart}s")
            # Map the sorted page_content back to the original document objects
            page_content_to_doc_map = {
                doc.page_content: doc for doc_list in doc_lists for doc in doc_list
            }
            logger.debug(f"rrf page_content_to_doc_map:{int(time.time()) - mainQueryStart}s")
            sorted_docs = [
                page_content_to_doc_map[page_content] for page_content in sorted_documents
            ]
            logger.debug(f"rrf sorted_docs:{int(time.time()) - mainQueryStart}s")
            print("sortlen:" + str(len(sorted_docs)))
            insert_table(sorted_docs[:10], input, index, query, query_kind, batch_id, fake_query)
            #insert_table(sorted_docs, input, index, query, query_kind, batch_id, fake_query)
            logger.debug(f"rrf insert_table:{int(time.time()) - mainQueryStart}s")

        case _:
            pass



def bge_ranking(query: str, doc_lists: list, is_ft: bool = True, top_k: int = 30, should_sort: bool = True):
    docs = []
    for doc in doc_lists:
        # print(len(doc_list))
        # for doc in doc_list:
        docs.append(doc)

    global g_model, g_tokenizer
    pairs = []
    for doc in docs:
        pair = [query, doc.page_content]
        pairs.append(pair)
    with torch.no_grad():
        inputs = g_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs.to(device)
        scores = g_model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = scores.cpu()
        float_scores = scores.numpy().tolist()
    logger.debug(f"rrf bge_ranking calc rank:{int(time.time()) - mainQueryStart}s")

    if len(float_scores) == len(docs):
        if should_sort:
            sorted_data = sorted(zip(docs, float_scores), key=lambda x: x[1], reverse=True)
            for i in sorted_data:
                doc_i = i[0]
                doc_i.metadata['rerank_score'] = i[1]
            sorted_docs, _ = zip(*sorted_data)
            return sorted_docs[:top_k]
        else:
            for index, doc in enumerate(docs):
                doc.metadata['rerank_score'] = float_scores[index]
            return docs[:top_k]

def calc_bge_score(query: str, doc_lists: list, is_ft: bool = True, top_k: int = 30):
    return bge_ranking(query, doc_lists, is_ft, top_k, False)

if __name__ == '__main__':
    start = int(time.time())

    #from api import create_app
    #app = create_app()

    query_kind = 1  # 1 law, 2 sec
    filename = 'law_1201_100queries.txt'
    user_id = 'test_lsl'

    """

    """

    top_k = 15

    # es_filter_example = [
    #     {"term": {"metadata.timeliness": '现行有效'}},
    #     {"term": {"metadata.is_tax.keyword": 1}}
    #     # {"term": {"metadata.is_tax.keyword": 1}}
    # ]

    # es_filter = []

    batch_id = create_batch(filename, 'law_test_0125', user_id)
    print("batch_id:", batch_id)
    input_file = "../data/{}".format(filename)
    inputs = read_file_lines(input_file)
    i = 0

    # h = json.loads(h)
    # index_list = [32,69,60,94,17,22,75,18,82,99,66,9,89,80]
    index_list = [32,69,94,22,75,18,82,66,9,80]

    logger.debug(f"rrf main start loading ESSearchers:{int(time.time()) - start}s")
    g_ESSearcher_in_main_thread = ESSearcher(query_kind)
    g_ESSearcher_in_child_thread = ESSearcher(query_kind)
    logger.debug(f"rrf main ESSearchers launched:{int(time.time()) - start}s")

    is_ft = False
    #pre load models
    logger.debug(f"rrf bge_ranking:{int(time.time()) - start}s")

    if is_ft:
        g_tokenizer = AutoTokenizer.from_pretrained(BGE_RERANK_FT_PATH)
        g_model = AutoModelForSequenceClassification.from_pretrained(BGE_RERANK_FT_PATH)
    else:
        g_tokenizer = AutoTokenizer.from_pretrained(BGE_RERANK_PATH)
        g_model = AutoModelForSequenceClassification.from_pretrained(BGE_RERANK_PATH)
    g_model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    g_model.to(device)
    logger.debug(f"rrf load rerank model:{int(time.time()) - start}s")

    for input in inputs:
        i += 1 # index
        mainQueryStart = int(time.time())
        # if i not in index_list:
        if i == 101:
            break
        question_to_query(input, query_kind, batch_id, i, '')
        logger.debug(f"finish i - {i}, input - {input}")
        logger.debug(f"Query {i} 总耗时:{int(time.time()) - mainQueryStart}s - batch_id:{batch_id}")


        #break;
    logger.debug(f"主查询耗时:{int(time.time()) - start}s")


    """
    历史打分相关
    """
    # update_unscored_hanlder([batch_id])

    logger.info(("Finished!!!!!!"))

    end = int(time.time())
    logger.debug(f"耗时:{end - start}s")
