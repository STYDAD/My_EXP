import subprocess
from multiprocessing import Process

def run_script(script_path):
    subprocess.run(["python", script_path])

def main():
    scripts = [
        'WS14/w14.py',
        'WS15/w15.py',
        'WS16/w16.py',
        'WS17/w17.py',
        'WS18/w18.py'
    ]

    processes = [Process(target=run_script, args=(script,)) for script in scripts]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()