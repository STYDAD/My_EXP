import subprocess
from multiprocessing import Process

def run_script(script_path):
    subprocess.run(["python", script_path])

def main():
    scripts = [
        '/WS19/w19.py',
        '/WS20/w20.py',
        '/WS21/w21.py',
        '/WS22/w22.py',
        '/WS23/w23.py'
    ]
    processes = [Process(target=run_script, args=(script,)) for script in scripts]

    for process in processes:
        process.start()
        
    for process in processes:
        process.join()

if __name__ == "__main__":
    main()