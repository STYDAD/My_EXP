import sys
import os
import logging
import pymongo
from Clean_importer import *
from pymongo.errors import AutoReconnect, OperationFailure
from tqdm import tqdm

from Tools import *

logging.basicConfig(filename='progress15.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 设置
year = '2015'
# Size = 10
# total_progress = 500
Size = 3000
total_progress = 9752556
PLAN_FILE =  year + 'Plan.txt'
DONE_FILE = year + 'Done.txt'
PATH = 'ws_' + year

# 检测文件单是否存在
def File_detect():
    try:
        with open(PLAN_FILE, 'r') as plan:
            pass
    except FileNotFoundError:
        with open(PLAN_FILE, 'w') as plan:
            all_files = sorted(os.listdir(PATH))
            for file_name in all_files:
                plan.write(f"{file_name},0,0\n")

    try:
        with open(DONE_FILE, 'r') as done:
            Update = []
            for line in done:
                status = line.split(',')[1][0]
                if status == '1':
                    Update.append(line)
        with open(DONE_FILE, 'w') as done:
            for line in Update:
                done.write(line = '\n')
    except FileNotFoundError:
        with open(DONE_FILE, 'w') as done:
            pass

# 返回一个需要处理的文件
def get_unprocessed_file():
    with lock:
        with open(PLAN_FILE, 'r') as plan, open(DONE_FILE, 'a+') as done:
            done.seek(0)
            done_lines = done.read().splitlines()
            done_lines = [i.split(',')[0] for i in done_lines]
            for line in plan:
                first_part = line.split(',')[0]
                if first_part not in done_lines:
                    done.write(first_part + ',0\n')
                    return first_part
    return None

# 文件处理
def process_file(file_name,pbar):
    WRIT = Writ(Size,year,file_name)
    WRIT.Initialize_Cursor()
    WRIT.Reload_JsonL_Index()
    Pro = WRIT.json_progress if WRIT.json_progress else 0
    while Pro < total_progress:
            try:
                start_time = time.time()
                WRIT.Modify_Jsonl()
                end_time = time.time()
            except Exception as e:
                print(f"Caught an unexpected exception: {e}\n")

            if not WRIT.Save_Json_Modified():
                print('\n',file_name,' 处理完成\n')
                break

            WRIT.Reload_JsonL_Index()
            elapsed_time = end_time - start_time
            pbar.set_postfix({"Data process time": elapsed_time})
            pbar.update(Size)
            Pro = WRIT.json_progress

# 每个线程
def worker(pbar):
    while True:
        file_name = get_unprocessed_file()
        if file_name is None:
            print(f"线程 {threading.current_thread().name} 处理完成，即将关闭.")
            break
        process_file(file_name,pbar)
        with lock:
            with open(DONE_FILE, 'a') as done:
                done.write(file_name + ',1\n')


def main():
    File_detect()
    Writ_Data = Writ(Size,year)
    Writ_Data.Initialize_Cursor()
    with tqdm(total=total_progress, desc="Processing") as pbar:
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)
        
        # 多线程操作
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, name=f"Thread-{i+1}", args=(pbar,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        
        logger.removeHandler(console_handler)
    print(year+'年文书'f"优化完成")


if __name__ == "__main__":
    main()