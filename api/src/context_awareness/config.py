import os
from dotenv import load_dotenv
from nlp import setup as nlp_setup

os.environ["API_URL"] = "http://localhost:3000/v1"
load_dotenv()
nlp_setup.init()
