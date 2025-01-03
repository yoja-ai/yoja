import vertexai
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
    ToolConfig,
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

tool = Tool(function_declarations=[yoja_retrieve_function, yoja_retrieve_additional_function])

def _generate_with_conf_and_retry(model, user_prompt_content, generation_config):
    for attempt in range(1, 4):
        try:
            return model.generate_content(user_prompt_content, generation_config=generation_config)
        except InternalServerError as ise:
            print(f"_generate_with_conf_and_retry: Caught InternalServerError. attempt {attempt}")
            time.sleep(attempt*5)
        except ResourceExhausted as re:
            print(f"_generate_with_conf_and_retry: Caught ResourceExhausted. attempt {attempt}")
            time.sleep(attempt*10)
    print(f"_generate_with_conf_and_retry: Failed")
    return None

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
        print(f"_extract_name_entities: response={response}")
        content = response.candidates[0].content.parts[0].text.strip()
        print(f"_extract_name_entities: content={content}")
        try:
            js = json.loads(content)
        except Exception as ex:
            print(f"gemini_client._extract_named_entities: caught {ex} decoding content {content}")
            jss = _extract_json(content)
            print(f"_extract_name_entities: jss={jss}")
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

def _generate_with_retry(model, vertex_messages, tools=None, tool_config=None):
    for attempt in range(1, 4):
        try:
            if tools and tool_config:
                return model.generate_content(vertex_messages, tools=tools, tool_config=tool_config)
            else:
                return model.generate_content(vertex_messages)
        except InternalServerError as ise:
            print(f"_generate_with_retry: Caught InternalServerError. attempt {attempt}")
            time.sleep(attempt*5)
        except ResourceExhausted as re:
            print(f"_generate_with_retry: Caught ResourceExhausted. attempt {attempt}")
            time.sleep(attempt*10)
    print(f"_generate_with_retry: Failed")
    return None

def chat_using_gemini_assistant(yoja_index, tracebuf:List[str],
                                    filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                    assistants_thread_id:str, chat_config:ChatConfiguration, messages,
                                    searchsubdir=None, toolprompts=None) -> Tuple[str, str]:
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index 
    Returns the tuple (output, thread_id).  Returns (None, None) on failure.
    """
    print(f"chat_using_gemini_assistant: Entered. tool={tool}, messages={messages}")
    tools=[tool]
    if len(messages) == 1:
        system_instruction="You are a helpful assistant. Help me using knowledge from the provided tool only. Do not use your own knowledge to fullfil my requests"
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
        system_instruction="You are a helpful assistant. Help me using the context provided here. Do not use the provided tool if you can answer using the context provided. Do not use your own knowledge to fulfill my requests"
        tool_config=ToolConfig(
            function_calling_config=ToolConfig.FunctionCallingConfig(
                # ANY mode forces the model to predict only function calls
                mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                # Allowed function calls to predict when the mode is ANY. If empty, any  of
                # the provided function calls will be predicted.
                allowed_function_names=["additional_info_for_any_question_I_may_have"],
            )
        )
    print(f"chat_using_gemini_assistant: system_instruction={system_instruction}")
    model = GenerativeModel(
                model_name=ASSISTANTS_MODEL,
                system_instruction=system_instruction
                )
    print(f"model={model}")
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
    print(f"vertex_messages={vertex_messages}")
    response = _generate_with_retry(model, vertex_messages, tools, tool_config)
    print(response)

    if response.candidates[0].content.parts and len(response.candidates[0].content.parts) > 0:
        if response.candidates[0].content.parts[0].function_call:
            fc = response.candidates[0].content.parts[0].function_call
            if 'question' in fc.args:
                tool_arg_question = fc.args['question']
            elif 'prompt' in fc.args:
                tool_arg_question = fc.args['prompt']
            else:
                tool_arg_question = vertex_messages[-1]['parts'][0]
            if fc.name == 'additional_info_for_any_question_I_may_have':
                context:str = get_context(yoja_index, tracebuf,
                            filekey_to_file_chunks_dict, chat_config, tool_arg_question,
                            False, True, searchsubdir=searchsubdir, calc_tokens=_calc_tokens,
                            extract_main_theme=_extract_main_theme,
                            extract_named_entities=_extract_named_entities)
            else: # all other functions, including info_for_any_question_I_may_have
                context:str = get_context(yoja_index, tracebuf,
                            filekey_to_file_chunks_dict, chat_config, tool_arg_question,
                            True, False, searchsubdir=searchsubdir, calc_tokens=_calc_tokens,
                            extract_main_theme=_extract_main_theme,
                            extract_named_entities=_extract_named_entities)
            print(f"{prtime()}: Tool output: context={context}")
            tracebuf.append(f"{prtime()}: Tool output: context={context[:64]}...")
            vertex_messages.append(response.candidates[0].content)
            vertex_messages.append(Content(parts=[Part.from_function_response(name=fc.name, response={'content': context})]))
            print(f"vertex_messages after get_context={vertex_messages}")
            response = _generate_with_retry(model, vertex_messages)
            print(response)
            if response.candidates[0].content.parts and len(response.candidates[0].content.parts) > 0 \
                                            and response.candidates[0].content.parts[0].text:
                return response.candidates[0].content.parts[0].text.strip(), "notused", \
                    llm_run_usage(response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)
        elif response.candidates[0].content.parts[0].text:
            return response.candidates[0].content.parts[0].text, "notused", \
                        llm_run_usage(response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)

    return None, None, None

