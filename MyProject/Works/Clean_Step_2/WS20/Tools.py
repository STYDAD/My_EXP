import time
import logging

# 数据处理函数
def Process(WRIT):
    start_time = time.time()  
    WRIT.Modify_Jsonl()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

# 时间转换
def format_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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