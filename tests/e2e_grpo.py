import subprocess
import time
import httpx
import sys
import os
import json
import signal

# --- CONFIGURATION ---
SCHEDULER_PORT = 8080
NODE_PORT = 50052
SCHEDULER_URL = f"http://localhost:{SCHEDULER_PORT}"
NODE_ADDR = f"localhost:{NODE_PORT}"

def wait_for_port(port, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if port == SCHEDULER_PORT:
                with httpx.Client() as client:
                    resp = client.get(f"http://localhost:{port}/v1/models")
                    if resp.status_code == 200:
                        return True
            else:
                import socket
                with socket.create_connection(("localhost", port), timeout=1):
                    return True
        except:
            pass
        time.sleep(1)
    return False

def wait_for_node_registration(timeout=120):
    print("Waiting for node to register with scheduler...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with httpx.Client() as client:
                resp = client.get(f"{SCHEDULER_URL}/v1/nodes")
                if resp.status_code == 200:
                    nodes = resp.json().get("nodes", [])
                    # nodes might be a list of strings like ["localhost:50052"]
                    if any(NODE_ADDR in str(n) for n in nodes):
                        print(f"Node {NODE_ADDR} registered!")
                        return True
        except:
            pass
        time.sleep(1)
    return False

def run_e2e_test():
    processes = []
    try:
        # 1. Start Scheduler
        scheduler_cmd = ["./scheduler/target/release/scheduler"]
        env = os.environ.copy()
        env["SCHEDULER_PORT"] = str(SCHEDULER_PORT)
        # Suppress noisy tokenizer loading info if needed, or just let it show
        scheduler_proc = subprocess.Popen(
            scheduler_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            preexec_fn=os.setsid
        )
        processes.append(scheduler_proc)
        
        print("Starting scheduler...")
        if not wait_for_port(SCHEDULER_PORT):
            print("Scheduler failed to start")
            return

        # 2. Start Node
        node_cmd = [
            "uv", "run", "mlx-impl/node.py",
            "--port", str(NODE_PORT),
            "--checkpoint", "Qwen/Qwen3-0.6B"
        ]
        node_proc = subprocess.Popen(
            node_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid
        )
        processes.append(node_proc)
        
        print("Starting node (this may take a while to load weights)...")
        if not wait_for_port(NODE_PORT, timeout=120):
            print("Node failed to start or port not reachable")
            return

        if not wait_for_node_registration(timeout=60):
            print("Node registration failed")
            return

        # 3. Perform GRPO Rollout
        print("\n--- GRPO ROLLOUT ---")
        rollout_req = {
            "batch_id": "e2e-rollout-1",
            "prompts": [
                {"prompt_id": "p1", "prompt": "User: What is 2+2?\nAssistant: "},
                {"prompt_id": "p2", "prompt": "User: Capital of France?\nAssistant: "}
            ],
            "group_size": 2, # Smaller G for faster test
            "temperature": 0.8,
            "max_tokens": 16
        }
        
        t0 = time.perf_counter()
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{SCHEDULER_URL}/v1/grpo/rollout", json=rollout_req)
        t1 = time.perf_counter()
        
        if resp.status_code != 200:
            print(f"Rollout failed: {resp.status_code} - {resp.text}")
            return
            
        rollout_data = resp.json()
        print(f"Rollout took {t1 - t0:.2f}s")
        assert rollout_data["batch_id"] == "e2e-rollout-1"
        assert len(rollout_data["results"]) == 2
        
        # 4. Perform GRPO Judge
        print("\n--- GRPO JUDGE ---")
        judge_req = {
            "batch_id": "e2e-judge-1",
            "rubric": "Score the following response based on correctness and conciseness.",
            "items": [
                {"item_id": "j1", "prompt": "Prompt: 2+2. Completion: It is 4."},
                {"item_id": "j2", "prompt": "Prompt: France. Completion: Paris."}
            ],
            "temperature": 0.0,
            "max_tokens": 8
        }
        
        t0 = time.perf_counter()
        with httpx.Client(timeout=60) as client:
            resp = client.post(f"{SCHEDULER_URL}/v1/grpo/judge", json=judge_req)
        t1 = time.perf_counter()
        
        if resp.status_code != 200:
            print(f"Judge failed: {resp.status_code} - {resp.text}")
            return
            
        judge_data = resp.json()
        print(f"Judge took {t1 - t0:.2f}s")
        assert judge_data["batch_id"] == "e2e-judge-1"
        assert len(judge_data["results"]) == 2
        
        print("\n--- E2E TEST PASSED ---")
        
        # Benchmark results
        print("\n--- BENCHMARK ---")
        total_rollout_tokens = 2 * 2 * 16
        print(f"Rollout Throughput: {total_rollout_tokens / (t1 - t0):.1f} tokens/s (aggregate)")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nCleaning up processes...")
        for p in processes:
            try:
                # Kill the process group
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception as e:
                print(f"Failed to kill process {p.pid}: {e}")

if __name__ == "__main__":
    run_e2e_test()
