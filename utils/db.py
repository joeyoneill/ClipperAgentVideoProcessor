# Firestore DB file

# imports
from dotenv import load_dotenv
from google.cloud import firestore
import os

# load ENV Vars
load_dotenv()

# DB Client Init
db = firestore.Client(
    project=os.environ["GCP_PROJECT_ID"],
    database=os.environ["FIRESTORE_DB_NAME"]
)