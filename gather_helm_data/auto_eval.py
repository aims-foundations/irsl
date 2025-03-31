import os
import subprocess
import psutil
import time
import torch
import socket
import random
import yaml
from huggingface_hub import list_repo_refs
import shutil

def run_server(cmd_string):
    try:
        server_process = subprocess.Popen(
            cmd_string,
            shell=True,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.STDOUT,
            # text=True
        )
        return server_process
    
    except Exception as e:
        print(f"Error starting server: {e}", flush=True)
        return None

def shutdown_server(process):
    try:
        process = psutil.Process(process.pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()

        # Verify process termination
        time.sleep(2)  # Allow OS cleanup
        if process.is_running():
            process.terminate()
            print(f"Forcefully terminated process {process.pid}.", flush=True)
        else:
            print("Process terminated successfully.", flush=True)

        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU memory cleared.", flush=True)
        
    except psutil.NoSuchProcess:
        print("Process already terminated.", flush=True)
        
    except Exception as e:
        print(f"Error shutting down server: {e}", flush=True)
        
def find_available_port(start, end):
    while True:
        port = random.randint(start, end)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))  # Try to bind to the port
                return port  # Port is available
            except OSError:
                continue  # Port is in use, try another one
            
if __name__ == "__main__":
    os.chdir('./src')
    available_cudas = "4,5,6,7"
    tensor_parallel_size = len(available_cudas.split(","))
    assert tensor_parallel_size in (1, 2, 4, 8)
    
    model_ckpt_folder = "/lfs/skampere1/0/yuhengtu/helm/src/pythia_ckpt"
    
    repo_id = "EleutherAI/pythia-6.9b"
    refs = list_repo_refs(repo_id)
    branches = refs.branches
    versions = [branch.name for branch in branches]
    versions.remove("main")
    versions = sorted(versions, key=lambda x: int(x.split("step")[1]))
    # versions = [version for version in versions if not os.path.exists("benchmark_output/runs/mmlu_pythia-6.9b-" + version)]
    # versions = ["step10000", "step100000"]
    
    benchmarks = ["classic", "lite", "mmlu"]
    # benchmarks = ["mmlu"]
    benchmark2arg = {
        "classic": {
            "conf_paths": "helm/benchmark/presentation/run_entries_reeval_classic_new.conf",
            "schema": "helm/benchmark/static/schema_classic.yaml",
            "max_eval_instances": 1000,
            "priority": 2
        },
        "lite": {
            "conf_paths": "helm/benchmark/presentation/run_entries_reeval_lite_noninstruction.conf",
            "schema": "helm/benchmark/static/schema_lite.yaml",
            "max_eval_instances": 1000,
            "priority": 2
        },
        "mmlu":{
            "conf_paths": "helm/benchmark/presentation/run_entries_mmlu.conf",
            "schema": "helm/benchmark/static/schema_mmlu.yaml",
            "max_eval_instances": 10000,
            "priority": 4
        }
    }
    
    for version in versions[123:]:
        print(f"#################{version}", flush=True)
        
        port = find_available_port(1000, 9999)
        
        # modify model_deployments.yaml
        with open('helm/config/model_deployments.yaml', 'r') as file:
            data = yaml.safe_load(file)
        
        data["model_deployments"][0]["client_spec"]["args"]["base_url"] = 'http://0.0.0.0:' + str(port) + '/v1/'

        with open('helm/config/model_deployments.yaml', 'w') as file:
            yaml.safe_dump(data, file)
        
        # run vllm server
        try:
            vllm_cmd_string = (
                "CUDA_VISIBLE_DEVICES=" + available_cudas +
                " python -m vllm.entrypoints.openai.api_server" +
                " --model EleutherAI/pythia-6.9b" +
                " --port " + str(port) +
                " --tensor-parallel-size " + str(tensor_parallel_size) +
                " --revision " + version +
                " --download-dir " + model_ckpt_folder
            )
            vllm_process = run_server(vllm_cmd_string)
            time.sleep(220)
        except Exception as error:
            print(f"vllm server error: {error}", flush=True)
            continue

        for benchmark in benchmarks:
            print(f"#################{benchmark}", flush=True)
            
            conf_paths =  benchmark2arg[benchmark]["conf_paths"]
            max_eval_instances = benchmark2arg[benchmark]["max_eval_instances"]
            schema = benchmark2arg[benchmark]["schema"]
            priority = benchmark2arg[benchmark]["priority"]
            suite_name = benchmark + "_pythia-6.9b-" + version
            
            # run helm-run
            helm_run_cmd_string = "helm-run" + \
                " --conf-paths " + conf_paths + \
                " --num-train-trials 1" + \
                " --max-eval-instances " + str(max_eval_instances) + \
                " --priority " +  str(priority) + \
                " --suite " + suite_name + \
                " --models-to-run EleutherAI/pythia-6.9b" + \
                " --disable-cache"
                
            
            _ = subprocess.run(
                helm_run_cmd_string,
                shell=True,
                # capture_output=True,
                # text=True
            )
            
            # run helm-summarize
            helm_summarize_cmd_string = "helm-summarize" + \
                " --schema " + schema + \
                " --suite "+ suite_name
                
            _ = subprocess.run(
                helm_summarize_cmd_string,
                shell=True,
                # capture_output=True,
                # text=True
            )
        
        # shut down vllm
        shutdown_server(vllm_process)
        
        # delete huggingface cache for model that we just used
        shutil.rmtree(model_ckpt_folder)
        print(f"Deleted folder: {model_ckpt_folder}", flush=True)