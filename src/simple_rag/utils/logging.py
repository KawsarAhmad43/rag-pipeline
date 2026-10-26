import logging
import os
from config import config

# Define log directory and file path
log_file = os.path.join(config.LOG_PATH, "simple_rag.log")

# Ensure the directory exists
os.makedirs(config.LOG_PATH, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
