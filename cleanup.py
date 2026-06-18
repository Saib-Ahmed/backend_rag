import shutil
import os

dirs_to_remove = ['final_rag', 'RAG_system', 'md_output', '.tmp.driveupload']
files_to_remove = ['start_backend.py', 'start_runpod.sh', 'terminal_logs.txt', 'unified_app.py', 'unified_db.py', 'unified_rag.db']

for d in dirs_to_remove:
    if os.path.exists(d):
        try:
            shutil.rmtree(d)
            print(f"Removed directory: {d}")
        except Exception as e:
            print(f"Error removing directory {d}: {e}")

for f in files_to_remove:
    if os.path.exists(f):
        try:
            os.remove(f)
            print(f"Removed file: {f}")
        except Exception as e:
            print(f"Error removing file {f}: {e}")
