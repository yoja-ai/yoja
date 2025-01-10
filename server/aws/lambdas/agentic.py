import sys
import os
import json
from documentchunk import DocumentType, DocumentChunkDetails, DocumentChunkRange
from chatconfig import ChatConfiguration, RetrieverStrategyEnum
from typing import Tuple, List, Dict, Any, Self
from utils import prtime, llm_run_usage

MAX_RETRIEVE_ATTEMPTS=2

if 'GCLOUD_PROJECTID' in os.environ:
    from gemini_client import generate, retrieve
    print(f"Gemini LLM configured")
elif 'OLLAMA_HOST' in os.environ:
    print(f"ollama_not supported")
    sys.exit(255)
elif 'OPENAI_API_KEY' in os.environ:
    print(f"openai not supported")
    sys.exit(255)
else:
    print(f"No LLM configured")
    sys.exit(255)

PLANNER_MESSAGE = (
    """You are a task planner. You will be given some information your job is to think step by step and enumerate the steps to complete a request from a given user, using the provided context to guide you.
    You will not execute the steps yourself, but provide the steps to a helper who will execute them. Make sure each step consists of a single operation, not a series of operations. The helper has the following capabilities:
    1. Search through a collection of documents provided by the user. These are the user's own documents and will likely have the exact information the user needs
    7. Synthesize, summarize and classify the information received.
    Please output the step using a properly formatted python dictionary and list. Respond only with the plan json as described below and no additional text. Here are a few examples:
    Example 1: 
    User query: Write a performance self-assessment for Joe, consisting of a high-level overview of achievements for the year, a listing of the business impacts for each of these achievements, a list of skills developed and ways he's collaborated with the team.
    Your response:
    ```{"plan": ["Query documents for all contributions involving Joe this year", "Quantify the business impact for Joe's contributions", "Enumerate the skills Joe has developed this year", "List several examples of how Joe's work has been accomplished via team collaboration", "Formulate the performance review based on collected information"]}```

    Example 2:
    User query: Give me two recipes that involve beans
    Your response:
    ```{"plan": ["Query documents for recipes that mention beans", "Summarize two of the recipes"]}```
    """
)

ASSISTANT_PROMPT_WITH_TOOL = (
    """You are an AI assistant.
    When you receive a message, figure out a solution and provide a final answer. The message will be accompanied with contextual information. Use the contextual information to help you provide a solution.
    Make sure to provide a thorough answer that directly addresses the message you received.
    The context may contain extraneous information that does not apply to your instruction. If so, just extract whatever is useful or relevant and use it to complete your instruction.
    When the context does not include enough information to complete the task, use your available tools to retrieve the specific information you need.
    When you are using knowledge and web search tools to complete the instruction, answer the instruction only using the results from the search; do no supplement with your own knowledge.
    Be persistent in finding the information you need before giving up.
    If the task is able to be accomplished without using tools, then do not make any tool calls.
    When you have accomplished the instruction posed to you, you will reply with the text: ##SUMMARY## - followed with an answer.
    Important: If you are unable to accomplish the task, whether it's because you could not retrieve sufficient data, or any other reason, reply only with ##TERMINATE##.

    # Tool Use
    You have access to the "info_for_any_question_I_may_have" tool. Only use this tool and do not attempt to use anything not listed - this will cause an error.
    Respond in the format: <function_call> {"name": function name, "arguments": dictionary of argument name and its value}. Do not use variables.
    Only call one tool at a time.
    When suggesting tool calls, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.
    """
)

ASSISTANT_PROMPT_NO_TOOL = (
    """You are an AI assistant.
    When you receive a message, figure out a solution and provide a final answer. The context for this message is provided at the end of this messages starting with ## Context. Use the contextual information to help you provide a solution.
    Make sure to provide a thorough answer that directly addresses the message you received.
    The context may contain extraneous information that does not apply to your instruction. If so, just extract whatever is useful or relevant and use it to complete your instruction.
    Answer the instruction only using the context; do no supplement with your own knowledge.
    Be persistent in finding the information you need before giving up.
    When you have accomplished the instruction posed to you, you will reply with the text: ##SUMMARY## - followed with an answer.
    Important: If you are unable to accomplish the task, whether it's because you could not retrieve sufficient data, or any other reason, reply only with ##TERMINATE##.
    """
)

CRITIC_SYSTEM_PROMPT = "You are a critic of responses from a language model. Your job is to evaluate the response from a language model against the task given to the language model and determine if the response satisfies the task. The language model is supplied with context from the user's personal and confidential documents. Brief answers with no explanation from the language model are acceptable and considered satisfactory."

CRITIC_PROMPT = (
    """The user message was {user_message} \nThe following is the response of the model. '{last_output}\n'
    If the output of the model completely satisfies the instruction, then reply with ##YES##.
    For example, if the instruction is to list companies that use AI, then the output contains a list of companies that use AI.
    If the output contains the phrase 'I'm sorry but...' then it is likely not fulfilling the instruction. \n
    If the output of the model does not properly satisfy the instruction, then reply with ##NO## and the reason why.
    For example, if the instruction was to list companies that use AI but the output does not contain a list of companies, or states that a list of companies is not available, then the output did not properly satisfy the instruction.
    If it does not satisfy the instruction, please think about what went wrong with the previous instruction and give me an explanation along with the text ##NO##.
    """
)

CHECK_CONTEXT_SYSTEM_PROMPT = "You are a critic of responses from a language model that is supplied with selected context from a users personal document collection. Your job is to evaluate the response from a language model against the task given to the language model and determine if the context selected for the question is relevant to the question."
CHECK_CONTEXT_PROMPT = (
    """The user message was {user_message} \nThe search query performed on the users personal document collection was as follows: '{search_query}'\n.
    The following was the context produced as a result of the search query '{context_str}'\n'
    The following was the response of the model. '{last_output}\n'
    If the output of the model indicates that the context provided is corret, then reply with ##CORRECT##.
    For example, if the instruction is to determine the frequency of oil changes for the user's car, and the context consists of passages from the car's user manual, then the output context is correct.
    If the output contains the phrase 'I'm sorry but...' then it is likely that the context is not correct. \n
    If the context is not correct, then reply with ##WRONG##
    For example, if the instruction is to determine the frequency of oil changes for the user's car, and the context consists of passages from a garage door opener manual, then the output context is wrong.
    If the context is wrong, please think and give me a new search query for searching the user's document collection along with the text ##WRONG##.
    """
)

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


def _get_json(text):
    try:
        return json.loads(text)
    except Exception as ex:
        jss = _extract_json(text)
        try:
            return json.loads(jss)
        except Exception as ex:
            print(f"_get_json: caught {ex}")
    return None

def agentic_chat(yoja_index, tracebuf:List[str],
                filekey_to_file_chunks_dict:Dict[str, List[DocumentChunkDetails]],
                assistants_thread_id:str, chat_config:ChatConfiguration, messages,
                searchsubdir=None, toolprompts=None) -> Tuple[str, str]:
    print(f"agentic_chat: Entered")
    prompt_tokens = 0
    completion_tokens = 0

    user_msg = messages[-1]['content']
    content, prtokens, cmptokens = generate(PLANNER_MESSAGE,
                        [{'role': 'user', 'content': user_msg}],
                        False)
    prompt_tokens += prtokens
    completion_tokens += cmptokens

    if content:
        if hasattr(content, 'text'):
            plan_dict = _get_json(content.text)
            print(f"agentic_chat: plan_dict={plan_dict}")
        else:
            print(f"agentic_chat: No text and hence no plan. Fail")
            return None, None, None

    plan_step = plan_dict['plan'][0]
    user_message = messages[-1]['content']
    content1, prtokens, cmptokens = generate(ASSISTANT_PROMPT_WITH_TOOL,
                [{'role': 'user', 'content': f"Perform this step '{plan_step}' for this user message '{user_message}'"}],
                True)
    prompt_tokens += prtokens
    completion_tokens += cmptokens
    if content1:
        if hasattr(content1, 'text'):
            assistant_response = _get_json(content1.text)
            print(f"agentic_chat: assistant_response={assistant_response}")
        elif hasattr(content1, "function_call"):
            function_call = content1.function_call
            if function_call.name != 'info_for_any_question_I_may_have':
                print(f"Error. Unknown function call {function_call.name}")
                return None, None, None
            if 'question' in function_call.args:
                tool_arg_question = function_call.args['question']
            elif 'prompt' in function_call.args:
                tool_arg_question = function_call.args['prompt']
            else:
                print(f"agentic_chat: Error. No question or prompt in args to function")
                return None, None, None
            for retrieve_attempt in range(MAX_RETRIEVE_ATTEMPTS):
                print(f"Retrieving with question={tool_arg_question}")
                context_str:str = retrieve(yoja_index, tracebuf,
                                        filekey_to_file_chunks_dict, chat_config,
                                        tool_arg_question, searchsubdir=searchsubdir,
                                        num_hits_multiplier=retrieve_attempt)
                if len(plan_dict['plan']) > 1:
                    plan_step = plan_dict['plan'][1]
                else:
                    plan_step = user_msg
                content2, prtokens, cmptokens = generate(ASSISTANT_PROMPT_NO_TOOL,
                    [{'role': 'user', 'content': f"Perform this step '{plan_step}' for this user message '{user_message}'. ## Context: {context_str}"}],
                    False)
                prompt_tokens += prtokens
                completion_tokens += cmptokens
                if content2 and hasattr(content2, 'text'):
                    c2 = content2.text.strip()
                    if c2.startswith("##SUMMARY##"):
                        c2 = c2[11:].strip()
                    elif c2.startswith('##TERMINATE##'):
                        c2 = "Sorry, no useful answers found"
                    content4, prtokens, cmptokens = generate(CHECK_CONTEXT_SYSTEM_PROMPT,
                        [{'role': 'user', 'content': CHECK_CONTEXT_PROMPT.format(user_message=user_msg, search_query=tool_arg_question,
                                                                            context_str=context_str, last_output=c2)}],
                        False)
                    prompt_tokens += prtokens
                    completion_tokens += cmptokens
                    c4 = content4.text.strip()
                    print(f"agentic_chat: check context response={c4}")
                    if c4.startswith("##WRONG##"):
                        print(f"Context was incorrect")
                        if retrieve_attempt == (MAX_RETRIEVE_ATTEMPTS - 1):
                            return c2, "notused", llm_run_usage(prompt_tokens, completion_tokens)
                        else:
                            nq_ind = c4.find("**New Search Query:**")
                            if nq_ind != -1:
                                tool_arg_question = c4[nq_ind + len("**New Search Query:**"):].strip()
                                continue
                    else:
                        print(f"Context was accurate. Now checking whether user question was answered")
                        content3, prtokens, cmptokens = generate(CRITIC_SYSTEM_PROMPT,
                            [{'role': 'user', 'content': CRITIC_PROMPT.format(user_message=user_msg, last_output=c2)}],
                            False, temperature=1.0)
                        prompt_tokens += prtokens
                        completion_tokens += cmptokens
                        print(f"agentic_chat: critic response={content3}")
                        if content3.text.find("##NO##") == -1:
                            return c2, "notused", llm_run_usage(prompt_tokens, completion_tokens)
                        else:
                            print(f"Context was accurate. However, question was not answered. Returning the answer nonetheless")
                            return c2, "notused", llm_run_usage(prompt_tokens, completion_tokens)
    return None, None, None
