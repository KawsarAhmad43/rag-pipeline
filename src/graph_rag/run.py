import os
import subprocess
import sys
import time
import logging
import socket
import psutil
import signal
import torch
from config import config
from utils.system import clean_gpu_cache

log_dir = config.LOG_PATH
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "run.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Set SO_REUSEADDR to ignore TIME_WAIT sockets from previous runs
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return False
        except socket.error:
            return True

def terminate_process_on_port(port: int):
    try:
        processes_terminated = False
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port:
                pid = conn.pid
                if pid:
                    try:
                        proc = psutil.Process(pid)
                        logger.warning(f"Killing process {pid} ({proc.name()}) using port {port}")
                        proc.kill()  # Use SIGKILL for immediate termination
                        proc.wait(timeout=2)
                        logger.info(f"Process {pid} killed successfully")
                        processes_terminated = True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

        if not processes_terminated:
            # Fallback to shell command if psutil didn't find/kill anything
            try:
                subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False, capture_output=True)
                logger.info(f"Attempted to free port {port} using fuser")
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Failed to terminate process on port {port}: {str(e)}")

def run_fastapi():
    port = 2000
    host = "0.0.0.0"
    logger.info(f"Checking if port {port} is available for FastAPI...")
    if is_port_in_use(port):
        logger.warning(f"Port {port} is in use, attempting to free it...")
        terminate_process_on_port(port)
        time.sleep(1)
        if is_port_in_use(port):
            logger.error(f"Port {port} still in use, cannot start FastAPI server")
            raise RuntimeError(f"Port {port} is already in use")

    # Set environment variable to disable RAGAs analytics
    os.environ['RAGAS_DO_NOT_TRACK'] = 'true'
    clean_gpu_cache()

    logger.info("Starting FastAPI server in 'graph' conda environment...")
    logger.info(f"Frontend will be available at: http://10.0.0.52:{port}/graph/")

    # Use conda run to ensure the correct environment is used
    cmd = [
        "conda", "run", "-n", "graph", "--no-capture-output",
        "uvicorn",
        "app.api.main:app",
        "--host", host,
        "--port", str(port),
        "--reload",
        "--reload-dir", "app",
        "--loop", "asyncio"
    ]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(__file__),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Give it a moment to start
        time.sleep(3)
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=10)
            logger.error(f"FastAPI process failed to start. Stdout: {stdout}, Stderr: {stderr}")
            raise RuntimeError(f"FastAPI server failed to start: {stderr}")

        logger.info(f"FastAPI server started successfully on http://{host}:{port}")
        return process
    except Exception as e:
        logger.error(f"Error starting FastAPI server: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting application...")
    fastapi_process = None
    try:
        fastapi_process = run_fastapi()
        while True:
            if fastapi_process and fastapi_process.poll() is not None:
                stdout, stderr = fastapi_process.communicate(timeout=10)
                logger.error(f"FastAPI process terminated unexpectedly. Stdout: {stdout}, Stderr: {stderr}")
                raise RuntimeError("FastAPI process crashed")
            time.sleep(1)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        clean_gpu_cache()
        if fastapi_process:
            try:
                fastapi_process.terminate()
                fastapi_process.wait(timeout=5)
            except Exception as term_error:
                logger.error(f"Error terminating FastAPI process: {str(term_error)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, terminating processes...")
        clean_gpu_cache()
        if fastapi_process:
            try:
                fastapi_process.terminate()
                fastapi_process.wait(timeout=5)
            except Exception as term_error:
                logger.error(f"Error terminating FastAPI process: {str(term_error)}")
        logger.info("Processes terminated successfully")
        sys.exit(0)
