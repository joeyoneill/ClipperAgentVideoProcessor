# GCP Cloud Storage Client File

# imports
from dotenv import load_dotenv
from google.cloud import storage
import os

# Load Env Vars
load_dotenv()

# Storage Client
gcs_client = storage.Client(project=os.environ["GCP_PROJECT_ID"])