import logging
import os
from datetime import datetime
from from_root import from_root

Log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", Log_file)

os.makedirs(logs_path, exist_ok=True)

Log_File_Path = os.path.join(logs_path, Log_file)
logging_str = "[%(asctime)s] : %(name)s : %(levelname)s : %(module)s : %(message)s"

logging.basicConfig(
    filename=Log_File_Path,
    format= logging_str,
    level= logging.DEBUG,
)