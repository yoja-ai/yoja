import google.generativeai as genai
import os
from typing import Tuple, List, Dict, Any, Self
import faiss_rm
from documentchunk import DocumentType, DocumentChunkDetails, DocumentChunkRange
from chatconfig import ChatConfiguration, RetrieverStrategyEnum
import json
from utils import prtime, llm_run_usage

#ASSISTANTS_MODEL="gemini-1.5-flash"
ASSISTANTS_MODEL="gemini-pro"

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

yoja_retrieve = genai.protos.Tool(
    function_declarations = [
        genai.protos.FunctionDeclaration(
            name = "info_for_any_question_I_may_have",
            #description = "Search confidential and private information and return relevant passages for the given question or search and return relevant passages that provide details of the mentioned subject",
            #description = "Search user's personal documents collection and provide relevant passages to answer the given question",
            #description = "Search and retrieve relevant documents from the my personal document collection",
            #description = "Access my relevant personal documents from my personal document collection",
            description = "Get information for any question I may have",
            parameters = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    'question': genai.protos.Schema(type=genai.protos.Type.STRING)
                },
                required=['question']
            )
        )
    ])

tool_config = genai.protos.ToolConfig(
    function_calling_config=genai.protos.FunctionCallingConfig(
        mode=genai.protos.FunctionCallingConfig.Mode.ANY,
        allowed_function_names=["info_for_any_question_I_may_have"],
    )
)

def _extract_main_theme(text):
    model = genai.GenerativeModel(ASSISTANTS_MODEL)
    gemini_messages = [{'role': 'user', 'parts': [f"Extract the main topic as a single word in the following sentence and return the result as a single word: {text}"]}]
    response = model.generate_content(gemini_messages)
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
    model = genai.GenerativeModel(ASSISTANTS_MODEL)
    gemini_messages = [{'role': 'user', 'parts': [f"Extract any named entities present in the sentence. Return a JSON object in this format: {{ 'entities': ['entity1', 'entity2'] }}, without any additional text or explanation. Particularly, do not include text before or after the parseable JSON: {text}"]}]
    response = model.generate_content(gemini_messages)
    if response.candidates[0].content.parts and len(response.candidates[0].content.parts) > 0 \
                                        and response.candidates[0].content.parts[0].text:
        content = response.candidates[0].content.parts[0].text.strip()
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
    model = genai.GenerativeModel(ASSISTANTS_MODEL)
    return model.count_tokens(prompt)

def chat_using_gemini_assistant(faiss_rms:List[faiss_rm.FaissRM], documents_list:List[Dict[str,str]],
                                    index_map_list:List[Tuple[str,str]], index_type, tracebuf:List[str],
                                    filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                                    assistants_thread_id:str, chat_config:ChatConfiguration, messages,
                                    searchsubdir=None, toolprompts=None) -> Tuple[str, str]:
    """
    documents is a dict like {fileid: finfo}; 
    index_map is a list of tuples: [(fileid, paragraph_index)];  the index into this list corresponds to the index of the embedding vector in the faiss index 
    Returns the tuple (output, thread_id).  Returns (None, None) on failure.
    """
    print(f"chat_using_gemini_assistant: Entered. faiss_rms={faiss_rms}. yoja_retrieve={yoja_retrieve}")
    model = genai.GenerativeModel(ASSISTANTS_MODEL, tools=yoja_retrieve, tool_config=tool_config)
    print(f"model={model}")
    gemini_messages = []
    gemini_messages.append({'role': 'user', 'parts': [messages[-1]['content']]})
    print(f"gemini_messages={gemini_messages}")
    response = model.generate_content(gemini_messages)
    print(response)

    if response.candidates[0].content.parts and len(response.candidates[0].content.parts) > 0 \
                                        and response.candidates[0].content.parts[0].function_call:
        fc = response.candidates[0].content.parts[0].function_call
        if fc.name == 'info_for_any_question_I_may_have':
            tool_arg_question = fc.args['question']
            context:str = get_context(faiss_rms, documents_list, index_map_list, index_type, tracebuf,
                                        filekey_to_file_chunks_dict, chat_config, tool_arg_question,
                                        True, False, searchsubdir=searchsubdir, calc_tokens=_calc_tokens,
                                        extract_main_theme=_extract_main_theme,
                                        extract_named_entities=_extract_named_entities)
            print(f"{prtime()}: Tool output: context={context}")
            tracebuf.append(f"{prtime()}: Tool output: context={context[:64]}...")
            gemini_messages.append({'role': 'user', 'parts': [
                    genai.protos.Part(function_response = genai.protos.FunctionResponse(name='info_for_any_question_I_may_have', response={'result': context}))
                ]})
            response = model.generate_content(gemini_messages)
            print(response)
            if response.candidates[0].content.parts and len(response.candidates[0].content.parts) > 0 \
                                                and response.candidates[0].content.parts[0].text:
                if response.usage_metadata:
                    return response.candidates[0].content.parts[0].text.strip(), "notused", \
                        llm_run_usage(response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)
                else:
                    return response.candidates[0].content.parts[0].text.strip(), "notused", llm_run_usage(0, 0)

    return None, None, None

