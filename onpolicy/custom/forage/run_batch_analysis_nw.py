import os
import subprocess
from multiprocessing import Pool, cpu_count

# Path to the parent directory containing the folders
parent_dir = "/home/nathanwu/zfish/onpolicy/custom/forage/results/rmappo-MultiAgentForagingEnv-check"
# Path to your script you want to run on each folder
script_paths = ["eval_ZFish.py", "preprocess_flatten.py"]


def run_script(folder_item):
    folder_path = os.path.join(parent_dir, folder_item)
    if os.path.isdir(folder_path):
        print(f"Running script on: {folder_path}", flush=True)

        for script_path in script_paths:
            try:
                subprocess.run(["python", script_path, f"{folder_path}"], check=True)
                return (folder_item, True)
            except subprocess.CalledProcessError:
                print(f"File not made for {folder_item} - running again", flush=True)
                try:
                    subprocess.run(["python", script_path, f"{folder_path}"], check=True)
                    return (folder_item, True)
                except subprocess.CalledProcessError:
                    print(f"FAILED AGAIN for {folder_item}", flush=True)
                    return (folder_item, False)
    else:
        print(f"OOPS: {folder_path} is not a directory", flush=True)
        return (folder_item, None)

if __name__ == "__main__":
    files_to_run = os.listdir(parent_dir)
    files_to_run = [item for item in files_to_run if os.path.isdir(os.path.join(parent_dir, item))]

    # You can set the number of workers to whatever you want (cpu_count() is safe)
    with Pool(processes=min(len(files_to_run), cpu_count())) as pool:
        results = pool.map(run_script, files_to_run)
    
    # Report results
    failures = [item for (item, result) in results if result is False]
    print("\nFAILED items after two tries:", failures)