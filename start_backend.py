import subprocess
import sys
import time
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Starting Unified API Gateway (Port 8001)...")
    gateway = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "unified_app:app", "--host", "0.0.0.0", "--port", "8001"],
        cwd=base_dir
    )
    
    print("Starting RAG V1 Engine (Port 8002)...")
    rag1 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"],
        cwd=os.path.join(base_dir, "RAG_system")
    )
    
    print("Starting RAG V2 Engine (Port 8003)...")
    rag2 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "final_rag.api:app", "--host", "0.0.0.0", "--port", "8003"],
        cwd=base_dir
    )

    print("\n--- All backends are running! ---")
    print("Press Ctrl+C to stop all servers gracefully.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down all backend servers...")
        gateway.terminate()
        rag1.terminate()
        rag2.terminate()
        gateway.wait()
        rag1.wait()
        rag2.wait()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
