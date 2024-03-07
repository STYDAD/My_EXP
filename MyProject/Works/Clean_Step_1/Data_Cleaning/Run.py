import os
import sys

def run_script(script_path):
    os.system(f"{sys.executable} {script_path}")

def main():
    script_paths = [
        'Clean_Run_1.py',
        'Clean_Run_2.py',
    ]
    
    # 依次运行每个脚本
    for script in script_paths:
        run_script(script)

if __name__ == "__main__":
    main()