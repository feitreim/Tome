import argparse
import os
import signal
import subprocess
import time

import httpx

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
                    if any(NODE_ADDR in str(n) for n in nodes):
                        print(f"Node {NODE_ADDR} registered!")
                        return True
        except:
            pass
        time.sleep(1)
    return False

def run_e2e_test(num_prompts: int, group_size: int, rollout_tokens: int, judge_items: int, max_inflight_rollouts: int):
    processes = []
    try:
        # 1. Start Scheduler
        scheduler_cmd = ["./scheduler/target/release/scheduler"]
        env = os.environ.copy()
        env["SCHEDULER_PORT"] = str(SCHEDULER_PORT)
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
            "--checkpoint", "Qwen/Qwen3-0.6B",
            "--max-inflight-rollouts", str(max_inflight_rollouts)
        ]
        node_proc = subprocess.Popen(
            node_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid
        )
        processes.append(node_proc)

        # Function to print output from a process
        def print_output(p, name):
            for line in iter(p.stdout.readline, ""):
                print(f"[{name}] {line.strip()}")

        import threading
        threading.Thread(target=print_output, args=(scheduler_proc, "SCHEDULER"), daemon=True).start()
        threading.Thread(target=print_output, args=(node_proc, "NODE"), daemon=True).start()

        print("Starting node (this may take a while to load weights)...")
        if not wait_for_port(NODE_PORT, timeout=120):
            print("Node failed to start or port not reachable")
            return

        if not wait_for_node_registration(timeout=60):
            print("Node registration failed")
            return

        # 3. Perform GRPO Rollout
        print(f"\n--- GRPO ROLLOUT (P={num_prompts}, G={group_size}) ---")
        prompts = [
            {"prompt_id": f"p{i}", "prompt": f"User: What is {i}+{i}?\nAssistant: "}
            for i in range(num_prompts)
        ]

        rollout_req = {
            "batch_id": "e2e-rollout-1",
            "prompts": prompts,
            "group_size": group_size,
            "temperature": 0.8,
            "max_tokens": rollout_tokens
        }

        t0 = time.perf_counter()
        with httpx.Client(timeout=300) as client:
            resp = client.post(f"{SCHEDULER_URL}/v1/grpo/rollout", json=rollout_req)
        t1 = time.perf_counter()

        if resp.status_code != 200:
            print(f"Rollout failed: {resp.status_code} - {resp.text}")
            return

        rollout_data = resp.json()
        print(f"Rollout took {t1 - t0:.2f}s")
        assert rollout_data["batch_id"] == "e2e-rollout-1"
        assert len(rollout_data["results"]) == num_prompts

        # 4. Perform GRPO Judge
        print(f"\n--- GRPO JUDGE (Items={judge_items}) ---")
        items = [
            {"item_id": f"j{i}", "prompt": f"Prompt: {i}+{i}. Completion: It is {i*2}."}
            for i in range(judge_items)
        ]

        judge_req = {
            "batch_id": "e2e-judge-1",
            "rubric": "Score the following response based on correctness and conciseness.",
            "items": items,
            "temperature": 0.0,
            "max_tokens": 8
        }

        t0_j = time.perf_counter()
        with httpx.Client(timeout=60) as client:
            resp = client.post(f"{SCHEDULER_URL}/v1/grpo/judge", json=judge_req)
        t1_j = time.perf_counter()

        if resp.status_code != 200:
            print(f"Judge failed: {resp.status_code} - {resp.text}")
            return

        judge_data = resp.json()
        print(f"Judge took {t1_j - t0_j:.2f}s")
        assert judge_data["batch_id"] == "e2e-judge-1"
        assert len(judge_data["results"]) == judge_items

        print("\n--- E2E TEST PASSED ---")

        # Benchmark results
        print("\n--- BENCHMARK ---")
        total_rollout_tokens = num_prompts * group_size * rollout_tokens
        print(f"rollout + ref_lps Throughput: {total_rollout_tokens / (t1 - t0):.1f} tokens/s (aggregate)")
        print(f"Judge Throughput: {judge_items / (t1_j - t0_j):.1f} items/s")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nCleaning up processes...")
        for p in processes:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception as e:
                print(f"Failed to kill process {p.pid}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E GRPO System Test")
    parser.add_argument("--num-prompts", type=int, default=8, help="Number of prompts (P)")
    parser.add_argument("--group-size", type=int, default=16, help="Rollouts per prompt (G)")
    parser.add_argument("--rollout-tokens", type=int, default=64, help="Tokens to generate per rollout")
    parser.add_argument("--judge-items", type=int, default=192, help="Number of items to judge")
    parser.add_argument("--max-inflight-rollouts", type=int, default=64, help="Max sequences for the node")

    args = parser.parse_args()

    run_e2e_test(
        num_prompts=args.num_prompts,
        group_size=args.group_size,
        rollout_tokens=args.rollout_tokens,
        judge_items=args.judge_items,
        max_inflight_rollouts=args.max_inflight_rollouts
    )
