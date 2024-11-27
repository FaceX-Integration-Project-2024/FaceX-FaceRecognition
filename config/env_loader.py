import os
from dotenv import load_dotenv

def load_env_variables():
    load_dotenv()
    return {
        "DB_URL": os.getenv("DB_URL"),
        "DB_KEY": os.getenv("DB_KEY"),
        "LOCAL": os.getenv("LOCAL")
    }
