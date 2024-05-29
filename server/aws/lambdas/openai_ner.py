import json
import os

import openai

from tenacity import retry, wait_random_exponential, stop_after_attempt

OPENAI_MODEL = 'gpt-3.5-turbo-0613'

labels = [
    "person",      # people, including fictional characters
    "fac",         # buildings, airports, highways, bridges
    "org",         # organizations, companies, agencies, institutions
    "gpe",         # geopolitical entities like countries, cities, states
    "loc",         # non-gpe locations
    "product",     # vehicles, foods, appareal, appliances, software, toys 
    "event",       # named sports, scientific milestones, historical events
    "work_of_art", # titles of books, songs, movies
    "law",         # named laws, acts, or legislations
    "language",    # any named language
    "date",        # absolute or relative dates or periods
    "time",        # time units smaller than a day
    "percent",     # percentage (e.g., "twenty percent", "18%")
    "money",       # monetary values, including unit
    "quantity",    # measurements, e.g., weight or distance
]

def system_message(labels):
    return f"""
You are an expert in Natural Language Processing. Your task is to identify common Named Entities (NER) in a given text.
The possible common Named Entities (NER) types are exclusively: ({", ".join(labels)})."""

def assisstant_message():
    return f"""
EXAMPLE:
    Text: 'In Germany, in 1440, goldsmith Johannes Gutenberg invented the movable-type printing press. His work led to an information revolution and the unprecedented mass-spread / 
    of literature throughout Europe. Modelled on the design of the existing screw presses, a single Renaissance movable-type printing press could produce up to 3,600 pages per workday.'
    {{
        "gpe": ["Germany", "Europe"],
        "date": ["1440"],
        "person": ["Johannes Gutenberg"],
        "product": ["movable-type printing press"],
        "event": ["Renaissance"],
        "quantity": ["3,600 pages"],
        "time": ["workday"]
    }}
--"""

def user_message(text):
    return f"""
TASK:
    Text: {text}
"""

def enrich_entities(text: str, label_entities: dict) -> str:
    #print(f"enrich_entities: Entered. text={text}, label_entities={label_entities}")
    #return json.dumps(label_entities)
    return label_entities

def generate_functions(labels: dict) -> list:
    return [
        {   
            "type": "function",
            "function": {
                "name": "enrich_entities",
                "description": "Enrich Text with Knowledge Base Links",
                "parameters": {
                    "type": "object",
                        "properties": {
                            "r'^(?:' + '|'.join({labels}) + ')$'": 
                            {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "additionalProperties": False
                },
            }
        }
    ]

class OpenAiNer():
    def __init__(self):
        self._client = openai.OpenAI()

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
    def __call__(self, text):
        messages = [
              {"role": "system", "content": system_message(labels=labels)},
              {"role": "assistant", "content": assisstant_message()},
              {"role": "user", "content": user_message(text=text)}
          ]

        try:
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
                tools=generate_functions(labels),
                tool_choice={"type": "function", "function" : {"name": "enrich_entities"}}, 
                temperature=0,
                frequency_penalty=0,
                presence_penalty=0,
            )
        except openai.APIConnectionError as e:
            print("Server connection error: {e.__cause__}")  # from httpx.
            raise
        except openai.RateLimitError as e:
            print(f"OpenAI RATE LIMIT error {e.status_code}: (e.response)")
            raise
        except openai.APIStatusError as e:
            print(f"OpenAI STATUS error {e.status_code}: (e.response)")
            raise
        except openai.BadRequestError as e:
            print(f"OpenAI BAD REQUEST error {e.status_code}: (e.response)")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
        response_message = response.choices[0].message
        available_functions = {"enrich_entities": enrich_entities}  
        function_name = response_message.tool_calls[0].function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message.tool_calls[0].function.arguments)
        return function_to_call(text, function_args)

    def convert_to_conversation(self, ner_result):
        rv = []
        if 'person' in ner_result and ner_result['person']:
            rv.append('this is regarding people ' + " ".join(ner_result['person']))
        if 'fac' in ner_result and ner_result['fac']:
            rv.append('this happened in ' + " ".join(ner_result['fac']))
        if 'org' in ner_result and ner_result['org']:
            rv.append('this is regarding the organization ' + " ".join(ner_result['org']))
        if 'gpe' in ner_result and ner_result['gpe']:
            rv.append('this is regarding the geopolitical entity ' + " ".join(ner_result['gpe']))
        if 'loc' in ner_result and ner_result['loc']:
            rv.append('this is regarding the location ' + " ".join(ner_result['loc']))
        if 'product' in ner_result and ner_result['product']:
            rv.append('this is regarding product ' + " ".join(ner_result['product']))
        if 'event' in ner_result and ner_result['event']:
            rv.append('this is regarding the event ' + " ".join(ner_result['event']))
        if 'work_of_art' in ner_result and ner_result['work_of_art']:
            rv.append('this is regarding the work of art ' + " ".join(ner_result['work_of_art']))
        if 'law' in ner_result and ner_result['law']:
            rv.append('this is regarding the law ' + " ".join(ner_result['law']))
        if 'language' in ner_result and ner_result['language']:
            rv.append('this is regarding language ' + " ".join(ner_result['language']))
        if 'date' in ner_result and ner_result['date']:
            rv.append('on date ' + " ".join(ner_result['date']))
        if 'time' in ner_result and ner_result['time']:
            rv.append('at time ' + " ".join(ner_result['time']))
        if 'percent' in ner_result and ner_result['percent']:
            rv.append('percent ' + " ".join(ner_result['percent']))
        if 'money' in ner_result and ner_result['money']:
            rv.append('money ' + " ".join(ner_result['money']))
        if 'quantity' in ner_result and ner_result['quantity']:
            rv.append('quantity ' + " ".join(ner_result['quantity']))
        return rv
