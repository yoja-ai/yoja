import io
import json
import os
import traceback
import tempfile
import uuid
import time
import base64
import zlib
import datetime
from urllib.parse import unquote
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils import respond

def getversion(event, context):
    operation = event['requestContext']['http']['method']
    return respond(None, res={'version': os.environ['LAMBDA_VERSION']})
