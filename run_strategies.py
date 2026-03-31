"""Start both arbitrage bot instances in parallel."""
import subprocess
import time
import sys
import os

def run_bot(env_file):
    """Launch a bot process with a specific env file."""
    # Cleanup potential stale locks
    if "dn" in env_file:
        lock = "polymarket_arbitrage/data/dn/bot.lock"
        hport = "18080"
    else:
        lock = "polymarket_arbitrage/data/dir/bot.lock"
        hport = "18081"
    
    if os.path.exists(lock):
        try:
            os.remove(lock)
            print(f"[*] Cleaned stale lock: {lock}")
        except Exception:
            pass

    print(f"[*] Starting bot with {env_file} on port {hport}...")
    # We use -u for unbuffered output to see logs in real-time
    process_env = {**os.environ, "DOTENV_FILE": env_file, "HEALTH_PORT": hport}
    return subprocess.Popen(
        [sys.executable, "-u", "polymarket_arbitrage/main.py"],
        env=process_env,
        cwd="."
    )

if __name__ == "__main__":
    try:
        # Launch both
        p1 = run_bot("polymarket_arbitrage/.env.dn")
        p2 = run_bot("polymarket_arbitrage/.env.dir")
        
        print("[!] Both bots are running. Press Ctrl+C to stop.")
        
        while True:
            # Check if processes are still alive
            if p1.poll() is not None:
                print("[X] Delta Neutral bot stopped unexpectedly.")
                break
            if p2.poll() is not None:
                print("[X] Directional bot stopped unexpectedly.")
                break
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("[*] Stopping bots...")
        p1.terminate()
        p2.terminate()
        print("[*] Bots stopped.")
