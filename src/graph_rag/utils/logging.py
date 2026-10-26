import logging
import os
from config import config

log_file = os.path.join(config.LOG_PATH, "graph_rag.log")
os.makedirs(config.LOG_PATH, exist_ok=True)

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
