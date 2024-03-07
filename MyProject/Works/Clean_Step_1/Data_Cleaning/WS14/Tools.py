import time

# 数据处理函数
def Process(WRIT):
    start_time = time.time()  
    WRIT.Get_Data()
    end_time_1 = time.time()  
    elapsed_time_1 = end_time_1 - start_time  
    start_time = time.time() 
    WRIT.Clone_table()
    WRIT.Head_Data_Cleaning()
    WRIT.Plaintiff_Defendant()
    WRIT.Main_Text_Cleaning()
    WRIT.Process_Pass()
    WRIT.save_to_jsonl()
    end_time_2 = time.time()
    elapsed_time_2 = end_time_2 - start_time
    return elapsed_time_1,elapsed_time_2

# 时间转换
def format_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"