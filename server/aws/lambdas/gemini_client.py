import vertexai
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
    ToolConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold
)
from google.api_core.exceptions import InternalServerError, ResourceExhausted
import os
from typing import Tuple, List, Dict, Any, Self
import faiss_rm
from documentchunk import DocumentType, DocumentChunkDetails, DocumentChunkRange
from chatconfig import ChatConfiguration, RetrieverStrategyEnum
import json
from utils import prtime, llm_run_usage
from yoja_retrieve import get_context
import time
import base64

ASSISTANTS_MODEL="gemini-1.5-flash"
#ASSISTANTS_MODEL="gemini-pro"

crs = base64.b64decode(os.environ['GCLOUD_CREDENTIALS']).decode('utf-8')
with open('/tmp/creds.json', 'w') as wfp:
    wfp.write(crs)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/tmp/creds.json'
vertexai.init(project=os.environ['GCLOUD_PROJECTID'], location="us-central1")

yoja_retrieve_function = FunctionDeclaration(
    name="info_for_any_question_I_may_have",
    description = "Get information for any question I may have",
    parameters={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "My question, for which this function will look up information"},
        },
        "required": ["question"],
    },
)
yoja_retrieve_additional_function = FunctionDeclaration(
    name="additional_info_for_any_question_I_may_have",
    description = "Get additional information for any question I may have. Only to be used if the function info_for_any_question_I_may_have cannot provide enough information",
    parameters={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "My question, for which this function will look up information"},
        },
        "required": ["question"],
    },
)

tool = Tool(function_declarations=[yoja_retrieve_function]) # , yoja_retrieve_additional_function])

def _generate_with_conf_and_retry(model, user_prompt_content, generation_config, tools=None, tool_config=None):
    for attempt in range(1, 4):
        try:
            if tools:
                return model.generate_content(user_prompt_content, generation_config=generation_config,
                                            tools=tools, tool_config=tool_config)
            else:
                return model.generate_content(user_prompt_content, generation_config=generation_config)
        except InternalServerError as ise:
            print(f"_generate_with_conf_and_retry: Caught InternalServerError. attempt {attempt}")
            time.sleep(attempt*5)
        except ResourceExhausted as re:
            print(f"_generate_with_conf_and_retry: Caught ResourceExhausted. attempt {attempt}")
            time.sleep(attempt*10)
    print(f"_generate_with_conf_and_retry: Failed")
    return None

def _generate_with_retry(model, vertex_messages, tools=None, tool_config=None, temperature=0):
    for attempt in range(1, 4):
        try:
            if tools and tool_config:
                return model.generate_content(vertex_messages,
                                generation_config=GenerationConfig(temperature=temperature),
                                tools=tools, tool_config=tool_config)
            else:
                return model.generate_content(vertex_messages,
                                generation_config=GenerationConfig(temperature=temperature))
        except InternalServerError as ise:
            print(f"_generate_with_retry: Caught InternalServerError. attempt {attempt}")
            time.sleep(attempt*5)
        except ResourceExhausted as re:
            print(f"_generate_with_retry: Caught ResourceExhausted. attempt {attempt}")
            time.sleep(attempt*10)
    print(f"_generate_with_retry: Failed")
    return None

def generate(system_instruction, messages, use_tools, temperature=0):
    print(f"generate: Entered. use_tools={use_tools}, system_instruction={system_instruction}, messages={messages}")
    vertex_messages = []
    for msg in messages:
        if msg['role'] == 'user':
            role = 'user'
        else:
            role = 'model'
        msgtext = msg['content']
        if 'source' in msg:
            for src in msg['source']:
                msgtext += f"\nsource is {src['name']}"
        vertex_messages.append(Content(role=role, parts=[Part.from_text(msgtext)]))

    model = GenerativeModel(
                model_name=ASSISTANTS_MODEL,
                system_instruction=system_instruction,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH
                }
            )
    if use_tools:
        tools=[tool]
        tool_config=ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                # ANY mode forces the model to predict only function calls
                mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                # Allowed function calls to predict when the mode is ANY. If empty, any  of
                # the provided function calls will be predicted.
                allowed_function_names=["info_for_any_question_I_may_have"],
            )
        )
    else:
        tools=None
        tool_config=None
    response = _generate_with_retry(model, vertex_messages, tools, tool_config, temperature=temperature)
    if response.candidates[0].content.parts and len(response.candidates[0].content.parts):
        return response.candidates[0].content.parts[0], response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count
    return None, 0, 0

def retrieve(yoja_index, tracebuf, filekey_to_file_chunks_dict,
                chat_config, tool_arg_question, searchsubdir, num_hits_multiplier=0):
        return get_context(yoja_index, tracebuf,
                    filekey_to_file_chunks_dict, chat_config, tool_arg_question,
                    True, False, searchsubdir=searchsubdir, calc_tokens=_calc_tokens,
                    extract_main_theme=_extract_main_theme,
                    extract_named_entities=_extract_named_entities,
                    num_hits_multiplier=num_hits_multiplier)

def _extract_main_theme(text):
    model = GenerativeModel(model_name=ASSISTANTS_MODEL)
    user_prompt_content = Content(
        role="user",
        parts=[
            Part.from_text(f"Extract the main topic as a single word in the following sentence and return the result as a single word: {text}"),
        ],
    )
    response = _generate_with_conf_and_retry(model,
        user_prompt_content,
        generation_config=GenerationConfig(temperature=0)
    )
    if response.candidates[0].content.parts and len(response.candidates[0].content.parts) > 0 \
                                        and response.candidates[0].content.parts[0].text:
        return response.candidates[0].content.parts[0].text.strip()
    return None
    

def _extract_json(text):
    lines = text.splitlines()
    code_array = []
    in_code = False
    for line in lines:
        if in_code:
            if line.strip() == '```':
                code_array.append("\n\n")
                in_code = False
            else:
                code_array.append(line)
        else:
            if line.strip() == '```json':
                in_code = True
    return '\n'.join(code_array)

def _extract_named_entities(text):
    model = GenerativeModel(model_name=ASSISTANTS_MODEL)
    user_prompt_content = Content(
        role="user",
        parts=[
            Part.from_text(
f'Extract any named entities present in the sentence. Return a parseable JSON object in this format: {{ "entities": ["entity1", "entity2"] }}, without any additional text or explanation. Particularly, do not include text before or after the parseable JSON: {text}'),
        ],
    )
    response = _generate_with_conf_and_retry(model,
        user_prompt_content,
        generation_config=GenerationConfig(temperature=0)
    )
    if response.candidates[0].content.parts and len(response.candidates[0].content.parts) > 0 \
                                        and response.candidates[0].content.parts[0].text:
        content = response.candidates[0].content.parts[0].text.strip()
        try:
            js = json.loads(content)
        except Exception as ex:
            jss = _extract_json(content)
            try:
                js = json.loads(jss)
            except Exception as ex:
                print(f"gemini_client._extract_named_entities: caught {ex}")
                return None
        if 'entities' in js and js['entities']:
            return js['entities']
    return None

def _calc_tokens(prompt):
    model = GenerativeModel(ASSISTANTS_MODEL)
    return model.count_tokens(prompt).total_tokens
