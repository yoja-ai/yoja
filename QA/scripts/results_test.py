import pandas as pd
import cryptography
from cryptography.fernet import Fernet
import requests
import json
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime, timezone
import os

# Configurations
ENCRYPTION_KEY = '<Enter Encryption Key>'
API_URL = 'https://chat.awsstaging4.yoja.ai/rest/v1/chat/completions'
EMAIL = '<Enter Email>'
FILE_PATH = 'processed_output.csv'
NUM_TRIES = 3  # number of times to ask each question
OPENAI_API_KEY = '<Enter Open AI API Key>'  # open ai api key
SIMILARITY_THRESHOLD = 0.85  # 85% similarity threshold to mark a response as a good response.

# Initialize encryption and OpenAI API key
fky = cryptography.fernet.Fernet(ENCRYPTION_KEY)
encrypted_email = fky.encrypt(EMAIL.encode())
cookie = f"yoja-user={encrypted_email.decode()}"
headers = {'Cookie': cookie, 'Content-Type': 'application/json'}
openai.api_key = OPENAI_API_KEY

#Generate unique log file names with timestamp
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
LOG_FILE_PATH = f'test_logs_{timestamp}.txt'
EXCEPTION_LOG_FILE_PATH = 'exception_log.txt'

def get_run_number():
    # exception log file is append only and will retrieve previous run number from the file while starting a new run, if run number isn't found we start afresh.
    """Retrieve the current run number from the exception log file."""
    if not os.path.exists(EXCEPTION_LOG_FILE_PATH):
        return 1
    with open(EXCEPTION_LOG_FILE_PATH, 'r') as f:
        lines = f.readlines()
    if lines:
        last_line = lines[-1]
        if last_line.startswith("Run") and "completed" in last_line:
            last_run_number = int(last_line.split()[1])
            return last_run_number + 1
    return 1

run_counter = get_run_number()

def log_exception(exception_message, run_number, question):
    """Log exceptions to the exception log file with a timestamp, run number, and question."""
    with open(EXCEPTION_LOG_FILE_PATH, 'a') as exception_log_file:
        exception_log_file.write(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')} - Run {run_number}\n")
        exception_log_file.write(f"Question: {question}\n")
        exception_log_file.write(f"{exception_message}\n")

def send_request(question):
    """Send question to the Yoja API multiple times and collect all responses."""
    postdata = {
        'messages': [{'role': 'user', 'content': question}],
        'model': 'gpt-3-5-turbo',
        'stream': True,
        'temperature': 1,
        'top_p': 0.7
    }
    responses = []
    for i in range(NUM_TRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=postdata)
            response.raise_for_status()
            # Extracting the actual content from the response
            resp = response.text
            json_str = resp[5:]
            resp_dict = json.loads(json_str)
            content = resp_dict['choices'][0]['delta']['content']
            responses.append(content)
            print(f"Response {i+1}: {responses[-1]}")
        except Exception as e:
            error_message = f"Error on request {i+1}: {str(e)}"
            responses.append(f"Error: {error_message}")
            log_exception(error_message, run_counter, question)
            print(error_message)
    return responses

def generate_embeddings(text):
    """Generate embeddings using OpenAI's embedding model."""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def calculate_similarity(responses, expected_answer):
    """Calculate semantic similarity for each response against the same expected answer."""
    expected_embedding = generate_embeddings(expected_answer)
    similarity_scores = []
    for i, response in enumerate(responses):
        if 'Error' not in response:
            response_embedding = generate_embeddings(response)
            score = cosine_similarity([expected_embedding], [response_embedding])[0][0]
            similarity_scores.append(score)
            print(f"Similarity score for response {i+1}: {score:.4f}")
        else:
            similarity_scores.append(None)
            print(f"Skipping similarity calculation for response {i+1} due to error.")
    return similarity_scores

def process_questions(data):
    """Process each question, send it multiple times, and evaluate responses."""
    global run_counter
    results = []
    tests_passed = 0
    total_tests = len(data)
    log_entries = []
    
    with open(EXCEPTION_LOG_FILE_PATH, 'a') as exception_log_file:
        exception_log_file.write(f"\n========== Run {run_counter} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')} ==========\n")

    for index, row in data.iterrows():
        print(f"\nProcessing question: {row['Question']}")
        log_entries.append(f"\nProcessing question: {row['Question']}\n{'-'*80}")
        responses = send_request(row['Question'])
        similarity_scores = calculate_similarity(responses, row['Extracted_Content'])
        
        positives = [score for score in similarity_scores if score is not None and score >= SIMILARITY_THRESHOLD]
        test_passed = len(positives) >= 2  # Test passes if 2 out of 3 responses are positive for each question
        if test_passed:
            tests_passed += 1
        
        log_entries.append(f"Responses:\n" + "\n".join([f"Response {i+1}: {resp}" for i, resp in enumerate(responses)]))
        log_entries.append(f"\nSimilarity Scores:\n" + "\n".join([f"Score {i+1}: {score:.4f}" if score is not None else f"Score {i+1}: Error" for i, score in enumerate(similarity_scores)]))
        log_entries.append(f"\nTest Passed: {test_passed}\n{'='*80}\n")
        
        results.append({
            'Question': row['Question'],
            'Responses': responses,
            'Expected Answer': row['Extracted_Content'],
            'Similarity Scores': similarity_scores,
            'Test Passed': test_passed
        })
    
    #calculate percentage of tests passed
    pass_percentage = (tests_passed / total_tests) * 100
    print(f"Percentage of tests passed: {pass_percentage:.2f}%")
    log_entries.append(f"\nPercentage of tests passed: {pass_percentage:.2f}%")
    
    #Save logs to file
    with open(LOG_FILE_PATH, 'w') as log_file:
        for entry in log_entries:
            log_file.write(entry + "\n")
    
    #recording completion of writing to exceptions file
    with open(EXCEPTION_LOG_FILE_PATH, 'a') as exception_log_file:
        exception_log_file.write(f"Run {run_counter} completed\n")

    return results

# Load data and process
data = pd.read_csv(FILE_PATH)
results = process_questions(data)
