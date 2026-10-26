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

# Create logs directory if it doesn't exist
log_dir = config.LOG_PATH
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "run.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_gpu_cache():
    """Clean the GPU cache using torch.cuda.empty_cache()."""
    if torch.cuda.is_available():
        logger.info("Cleaning GPU cache...")
        torch.cuda.empty_cache()
        logger.info("GPU cache cleaned successfully")
    else:
        logger.warning("No CUDA device available, skipping GPU cache cleaning")

def is_port_in_use(port: int) -> bool:
    """Check if the specified port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return False
        except socket.error:
            return True

def terminate_process_on_port(port: int):
    """Attempt to terminate any process using the specified port."""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            for conn in proc.info.get('connections', []):
                if conn.laddr.port == port:
                    logger.warning(f"Terminating process {proc.info['pid']} ({proc.info['name']}) using port {port}")
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=5)
                    logger.info(f"Process on port {port} terminated successfully")
                    return
    except Exception as e:
        logger.error(f"Failed to terminate process on port {port}: {str(e)}")

def run_fastapi():
    """Start the FastAPI server after checking port availability."""
    port = 9000
    logger.info(f"Checking if port {port} is available for FastAPI...")
    if is_port_in_use(port):
        logger.warning(f"Port {port} is in use, attempting to free it...")
        terminate_process_on_port(port)
        time.sleep(1)  # Wait briefly to ensure port is freed
        if is_port_in_use(port):
            logger.error(f"Port {port} still in use, cannot start FastAPI server")
            raise RuntimeError(f"Port {port} is already in use")

    clean_gpu_cache()  # Clean GPU cache before starting FastAPI
    logger.info("Starting FastAPI server...")
    cmd = [
        "uvicorn",
        "app.api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--reload",
        "--reload-dir",
        "app",
    ]
    try:
        process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(__file__),
            # stdout=subprocess.PIPE, # Inherit stdout/stderr to show logs in docker
            # stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(3)  # Wait for process to stabilize
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=10)
            logger.error(f"FastAPI process failed to start. Stdout: {stdout}, Stderr: {stderr}")
            raise RuntimeError(f"FastAPI server failed to start: {stderr}")
        logger.info(f"FastAPI server started successfully on http://localhost:{port}")
        return process
    except Exception as e:
        logger.error(f"Error starting FastAPI server: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting application...")
    fastapi_process = None
    try:
        fastapi_process = run_fastapi()

        # Keep the script running and monitor the FastAPI process
        while True:
            if fastapi_process and fastapi_process.poll() is not None:
                stdout, stderr = fastapi_process.communicate(timeout=10)
                logger.error(f"FastAPI process terminated unexpectedly. Stdout: {stdout}, Stderr: {stderr}")
                raise RuntimeError("FastAPI process crashed")
            time.sleep(1)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        clean_gpu_cache()  # Clean GPU cache on error
        if fastapi_process:
            try:
                fastapi_process.terminate()
                fastapi_process.wait(timeout=5)
            except Exception as term_error:
                logger.error(f"Error terminating FastAPI process: {str(term_error)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, terminating processes...")
        clean_gpu_cache()  # Clean GPU cache on Ctrl+C
        if fastapi_process:
            try:
                fastapi_process.terminate()
                fastapi_process.wait(timeout=5)
            except Exception as term_error:
                logger.error(f"Error terminating FastAPI process: {str(term_error)}")
        logger.info("Processes terminated successfully")
        sys.exit(0)
