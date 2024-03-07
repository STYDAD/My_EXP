import sys
import os
import logging
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# sys.path.append(parent_dir)
from Clean_importer import *
from Tools import *



logging.basicConfig(filename='progress15.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def main():
    # 设置
    year = '2015'
    
    # Size = 10
    # total_progress = 1050
    
    Size = 3000
    total_progress = 9752556

    start_time_0 = time.time()
    Writ_Data = Writ(Size,year)

    elapsed_time_1 = 0
    elapsed_time_2 = 0
    Writ_Data.RELOAD() 
    Writ_Data.Initialize_Cursor()

    LOAD = 0
    logging.info(f"Cleaning start, Batch_id:{year}\n")
    with tqdm(total=total_progress, desc="Processing") as pbar:
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)
        while Writ_Data.Progress < total_progress:
            try:
                elapsed_time_1,elapsed_time_2 = Process(Writ_Data)
            except (pymongo.errors.NetworkTimeout, AutoReconnect, OperationFailure) as e:
                logging.info(f"Network Error Occured: {e},\n")
                print(f"Caught a NetworkTimeout exception: {e}\n")
                Writ_Data.save_error_index()
                try:
                    # 尝试10次重连，将这次的坏数据index保存
                    reconnect_with_retry(Writ_Data.ReLogin, max_retries=30, delay_seconds=1)
                    elapsed_time_1,elapsed_time_2 = Process(Writ_Data)
                except ReconnectionError as e:
                    logging.info(f"Network Clashed: {e},\n")
                    print(f"Error: {e}")
                    break
            except Exception as e:
                # 捕获其他异常
                logging.info(f"Caught an unexpected exception: {e},\n")
                print(f"Caught an unexpected exception: {e}\n")
                # traceback.print_exc()
                Writ_Data.save_error_index()
                
            if Writ_Data.IfEnd:
                print('清洗完成')
                break
            if LOAD == 0:
                
                pbar.update(Writ_Data.Progress)
                pbar.update(Size)
            else:
                pbar.update(Size)
            LOAD += 1
            if LOAD % 50 == 0:

                Time_Step = round(time.time() - start_time_0)
                Time_Cur = format_seconds(Time_Step)
                Time_Tol = format_seconds(round((total_progress) / Writ_Data.Progress * Time_Step))
                Progress_Cur = round(100 * Writ_Data.Progress / total_progress)
                Speed = round(Writ_Data.Progress / Time_Step,2)
                logging.info(f"50 Batch Completed. Processing:{Progress_Cur}%.[ {Time_Cur}/{Time_Tol}, {Speed}it/s ]\n")
            pbar.set_postfix({"Data retrieval time": elapsed_time_1, "Data process time": elapsed_time_2})
        logger.removeHandler(console_handler)
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time_0
    logging.info(f"CLeaning Completed. Time-Consuming: {format_seconds(round(elapsed_time))}.\n")
    print(f"运行时间：{format_seconds(round(elapsed_time))}")


if __name__ == "__main__":
    main()