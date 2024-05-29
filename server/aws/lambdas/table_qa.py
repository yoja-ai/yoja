import dataclasses
import openai
import openai.types.chat.chat_completion
import pprint
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import regex
import pandasql
import io
import traceback_with_variables
import os

# print_exc() return value signature is 'no return': to avoid vscode marking code as unreachable, use 'if'
if len(__name__): traceback_with_variables.global_print_exc()

# https://emacs.stackexchange.com/questions/42137/ein-width-of-output-cells-in-notebook
# pd.set_option("display.width", 160)

os.environ["PYTHON_UNBUFFERED"] = "True"

def _chat_api_messages_to_prompt(messages:List[Dict]):
    raise Exception("Not implemented yet")

@dataclasses.dataclass
class TableQARequest:
    table_delimited_str:str
    question:str 
    delimiter:str = "|"
    model:str = "gpt-4"
    temperature:Optional[float] = None

# temperature" : number or null : Optional; Defaults to 1;
# What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
# We generally recommend altering this or top_p but not both.
# 
# top_p: number or null: Optional; Defaults to 1; 
# An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
# We generally recommend altering this or temperature but not both.
# 
# n: integer or null: Optional; Defaults to 1
# How many chat completion choices to generate for each input message. Note that you will be charged based on the number of generated tokens across all of the choices. Keep n as 1 to minimize costs.
# 
def _gpt_api_caller( messages:List[Dict], engine="gpt-4-1106-preview", verbose:bool=False, temperature=0.2, top_p=1, max_tokens=512, n=1 ) -> dict:
    """ invokes the gpt model (either text generation or the chat model).  engine=gpt-4-1106-preview|gpt-4|gpt-4-32k. Returns the model output"""
    # chat models
    # 'gpt-4', 
    # 'gpt-4-0314', 
    # 'gpt-4-32k', 
    # 'gpt-4-32k-0314', 
    # 'gpt-3.5-turbo', 
    # 'gpt-3.5-turbo-0301']:
    #
    #InvalidRequestError: The model `code-davinci-002` has been deprecated, learn more here: https://platform.openai.com/docs/deprecations.  Replace with gpt-4 or gpt-3.5-turbo-instruct
    #engine = "code-davinci-002"
    #engine = "gpt-4"
    #engine = "gpt-3.5-turbo-instruct"
    #prompt, 
    # https://platform.openai.com/docs/api-reference/chat/create
    # suffix: only for model fine tuning and not for chat api; string or null: Optional; Defaults to null; A string of up to 18 characters that will be added to your fine-tuned model name.
    # For example, a suffix of "custom-model-name" would produce a model name like ft:gpt-3.5-turbo:openai:custom-model-name:7p4lURel
    suffix=None
    stream=False
    logprobs=None
    # https://platform.openai.com/docs/api-reference/chat/create
    # stop: string / array / null: Optional: Defaults to null
    # Up to 4 sequences where the API will stop generating further tokens    
    stop=['```.', '``` ']
    presence_penalty=0
    frequency_penalty=0
    # best_of: applies to text completion model; integer or null; Optional; Defaults to 1
    # Generates best_of completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed.
    # When used with n, best_of controls the number of candidate completions and n specifies how many to return – best_of must be greater than n.
    # Note: Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for max_tokens and stop.
    best_of=1
    debug=False
    # Not defined in the API; not used below.
    prompt_end='\n\n'

    if verbose:
        print("chat api input: 'messages': ")
        pprint.pprint(messages)

    # InvalidRequestError: The model `code-davinci-002` has been deprecated, learn more here: https://platform.openai.com/docs/deprecations.  Replace with gpt-4 or gpt-3.5-turbo-instruct
    # engine = "code-davinci-002"
    #
    # https://help.openai.com/en/articles/7127956-how-much-does-gpt-4-cost : gpt-4-1106-preview, gpt-4, gpt-4-32k
    if engine in ( "gpt-4-1106-preview", "gpt-4", "gpt-4-32k"):
        print(f"Invoking chat api and getting its response: ")
        # pprint.pprint(messages)
        # since this is a chat model (gpt-4, gpt-3.5), use openai.ChatCompletion api..
        # https://github.com/openai/openai-python/discussions/742: openai v1.0.0 migration guide
        output_obj:openai.types.chat.ChatCompletion = openai.chat.completions.create(    
            model = engine, 
            messages = messages, 
            # suffix=suffix, 
            max_tokens=max_tokens, 
            temperature=temperature,
            top_p=top_p, 
            n=n,
            stream=stream,
            # logprobs=logprobs,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            # best_of=best_of,
        )
        output:dict = output_obj.model_dump()
        # output={
            #   "choices": [
            #     {
            #       "finish_reason": "stop",
            #       "index": 0,
            #       "logprobs": null,
            #       "message": {
            #         "content": "SQL: ```SELECT MAX(Year) FROM DF WHERE League='USL A-League';",
            #         "role": "assistant"
            #       },
            #       "text": "SQL: ```SELECT MAX(Year) FROM DF WHERE League='USL A-League';"
            #     }
            #   ],
            #   "created": 1710951470,
            #   "id": "chatcmpl-94svO7dcbERhzyM9H20cEcQw1DNvR",
            #   "model": "gpt-4-0613",
            #   "object": "chat.completion",
            #   "system_fingerprint": null,
            #   "usage": {
            #     "completion_tokens": 18,
            #     "prompt_tokens": 1405,
            #     "total_tokens": 1423
            #   }
            # }
            # output={
            #   "choices": [
            #     {
            #       "finish_reason": "stop",
            #       "index": 0,
            #       "logprobs": null,
            #       "message": {
            #         "content": "Python: ```DF[DF['port']=='Auckland'].sort_values(by='speed', ascending=False).iloc[0]['name']",
            #         "role": "assistant"
            #       },
            #       "text": "Python: ```DF[DF['port']=='Auckland'].sort_values(by='speed', ascending=False).iloc[0]['name']"
            #     }
            #   ],
            #   "created": 1710943509,
            #   "id": "chatcmpl-94qqzt0LZG2QRTBUm91FXS0vJdS1X",
            #   "model": "gpt-4-0613",
            #   "object": "chat.completion",
            #   "system_fingerprint": null,
            #   "usage": {
            #     "completion_tokens": 27,
            #     "prompt_tokens": 1794,
            #     "total_tokens": 1821
            #   }
            # }        
        output['choices'][0]['text'] = output['choices'][0]['message']['content'].strip('```')
        if 'Answer:' in output['choices'][0]['text'] and 'Answer: ```' not in output['choices'][0]['text']:
            output['choices'][0]['text'] = output['choices'][0]['text'].replace('Answer:', 'Answer: ```')
        print(f"engine={engine}; max_tokens={max_tokens}; temperature={temperature} (0 to 2); top_p={top_p} (0 to 1); n={n}, output={output}")
        return output
    #engine = "gpt-3.5-turbo-instruct" : no SQL or Python completion happens as it isn't a codex model
    elif engine in ( "gpt-3.5-turbo-instruct", "code-davinci-002" ):
        prompt = _chat_api_messages_to_prompt(messages)
        print(f"Invoking completion api and getting its response: prompt=\n{prompt}")
        output = openai.Completion.create(    
            engine = engine, 
            prompt = prompt, 
            suffix=suffix, 
            max_tokens=max_tokens, 
            temperature=temperature,
            top_p=top_p, 
            n=n,
            stream=stream,
            logprobs=logprobs,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of=best_of,
        )
        print(f"output={output}")
        return output
    else:
        raise Exception(f"Error: Unknown engine={engine}")

def _serialize_df(df:pd.DataFrame) -> str:
    # [HEAD]: by_race|white|black|aian*|asian|nhpi*
    # ---
    # [ROW] 1: 2000 (total population)|75.43%|4.46%|19.06%|5.24%|0.88%
    # [ROW] 2: 2000 (Hispanic only)|3.42%|0.33%|0.45%|0.16%|0.06%
    # [ROW] 3: 2005 (total population)|74.71%|4.72%|18.77%|5.90%|0.88%
    # ...``
    # [ROW] 6: Growth 2000–05 (non-Hispanic only)|3.49%|11.30%|4.02%|18.96%|5.86%
    # [ROW] 7: Growth 2000–05 (Hispanic only)|33.56%|21.02%|14.52%|27.89%|-1.95%
    table_rows:List[str] = []
    # first add the column header
    table_rows.append(f"[HEAD]: {'|'.join(df.columns.to_list())}")
    table_rows.append(f"---")
    for i in range(df.shape[0]):
        # TypeError: sequence item 0: expected str instance, numpy.int64 found
        table_rows.append(f"[ROW] {i+1}: {'|'.join( [ str(elem) for elem in df.iloc[i].tolist() ] )}")
    
    return "\n".join(table_rows)

def _exec_python(DF:pd.DataFrame, python_frag:str) -> None:
    """ Note: the dataframe argument must be DF, since the name DF is used in the few shot examples and in the prompt.  So the model generated python code will use DF.  Also assuming that the passed 'DF' dataframe is modified in place by the generated python code """
    print(f"Executing python=\n{python_frag} over dataframe: \n{DF}")
    loc:dict = locals()
    # TODO: why does locals() work?
    exec(python_frag, locals(), locals()) 
    print(f"After Executing python, modified dataframe: \n{loc['DF']}")
    return loc['DF']

def _exec_sql(DF:pd.DataFrame, sql:str) -> pd.DataFrame:
    """ Note: the dataframe argument must be DF, since the name DF is used in the few shot examples and in the prompt.  So the model generated python code will use DF """
    print(f"Executing sql={sql} over dataframe: \n{DF}")
    new_df:pd.DataFrame = pandasql.sqldf(sql)
    print(f"After Executing sql, the new dataframe is: \n{new_df}")
    return new_df

def iterate_prompt_exec_until_ans(engine:str, df_str:str, delimiter:str, df:pd.DataFrame, question:str, few_shot_examples:List[Dict] = None, prompt_template:str = None, max_hops=6, verbose:bool=True) -> Tuple[str, List[Dict]]:
    """ either df_str and delimiter must specified or df must be specified but not both.  'prompt_template', if specified, must have 'df' and 'question' variables in the template.  Returns a tuple of (answer, chat_messages)"""    
    
    if not (df_str and delimiter) or df:
        raise Exception(f"Either 'df_str' or 'df' must be specified: Neither was specified: df_str={df_str}; df={df}")
    
    # covert df_str to dataframe
    if df_str and delimiter: df = pd.read_csv(io.StringIO(df_str), sep=delimiter)
    
    if not prompt_template:
        prompt_template = \
"""The database table DF is shown as follows:
{df}

Answer the following question based on the data above: "{question}". Execute SQL or Python code step-by-step and finally answer the question. Choose from generating a SQL, Python code, or directly answering the question.
"""

    if not few_shot_examples:
        chat_api_user:str = "user"; chat_api_assistant = "assistant"
        few_shot_examples = [ 
{ 
"content": prompt_template.format(df="""[HEAD]: member|party|term
---
[ROW] 1: John Ryan|None|1859-1864
[ROW] 2: James Martin|None|1864-1869
[ROW] 3: James Watson|None|1869-1880
...
[ROW] 16: Member|Party|Term
[ROW] 17: Ian Armstrong|National|1981-2007""", question="which member served the longest?"),
"role": chat_api_user
},
{ 
"content": """
Python: ```
def get_duration(s):
    start = int(s.split('-')[0])
    end = int(s.split('-')[1])
    return end - start
DF['duration'] = DF.apply(lambda x: get_duration(x['term']), axis=1)
```.""",
"role": chat_api_assistant
},
{ 
"content": prompt_template.format(df="""[HEAD]: member|party|term|duration
---
[ROW] 1: John Ryan|None|1859-1864|5.0
[ROW] 2: James Martin|None|1864-1869|5.0
[ROW] 3: James Watson|None|1869-1880|11.0
...
[ROW] 16: Member|Party|Term|nan
[ROW] 17: Ian Armstrong|National|1981-2007|26.0""", question="which member served the longest?"),
"role": chat_api_user
},
{
"content": """SQL: ```SELECT member FROM DF ORDER BY duration DESC LIMIT 1;```.""", 
"role": chat_api_assistant
},
{
"content": prompt_template.format(df="""[HEAD]: member
---
[ROW] 1: Ian Armstrong""", question="which member served the longest?"),
"role": chat_api_user
},
{ 
"content": """Answer: ```Ian Armstrong```.""",
"role": chat_api_assistant
},
{ 
"content": prompt_template.format(df="""[HEAD]: number|builder|entered_service|withdrawn
---
[ROW] 1: 439|NZR Addington|9-1-1909|2-3-1957
[ROW] 2: 443|NZR Addington|1-12-1909|2-3-1957
[ROW] 3: 444|NZR Addington|3-2-1910|2-3-1957
[ROW] 4: 446|NZR Addington|30-4-1910|6-12-1946
[ROW] 5: 588|NZR Addington|14-10-1913|2-3-1957
[ROW] 6: 589|NZR Addington|11-11-1949|6-1949""", question="how many number were in service in 1910?"), 
"role": chat_api_user 
},
{
"content":"""Python: ```
def get_date_year(s):
    from dateutil import parser
    return parser.parse(s).year
DF['entered_service_year'] = DF.apply(lambda x: get_date_year(x['entered_service']), axis=1)
DF['withdrawn_year'] = DF.apply(lambda x: get_date_year(x['withdrawn']), axis=1)
```.""",
"role": chat_api_assistant 
},
{
"content":prompt_template.format(df="""[HEAD]: number|builder|entered_service|withdrawn|entered_service_year|withdrawn_year
---
[ROW] 1: 439|NZR Addington|9-1-1909|2-3-1957|1909|1957
[ROW] 2: 443|NZR Addington|1-12-1909|2-3-1957|1909|1957
[ROW] 3: 444|NZR Addington|3-2-1910|2-3-1957|1910|1957
[ROW] 4: 446|NZR Addington|30-4-1910|6-12-1946|1910|1946
[ROW] 5: 588|NZR Addington|14-10-1913|2-3-1957|1913|1957
[ROW] 6: 589|NZR Addington|11-11-1949|6-1949|1949|1949""", question="how many number were in service in 1910?"),
"role": chat_api_user
},
{
"content": """SQL: ```SELECT * FROM DF WHERE entered_service_year<=1910 AND withdrawn_year>=1910;```.""",
"role": chat_api_assistant
},
{
"content": prompt_template.format(df="""[HEAD]: number|builder|entered_service|withdrawn|entered_service_year|withdrawn_year
---
[ROW] 1: 439|NZR Addington|9-1-1909|2-3-1957|1909|1957
[ROW] 2: 443|NZR Addington|1-12-1909|2-3-1957|1909|1957
[ROW] 3: 444|NZR Addington|3-2-1910|2-3-1957|1910|1957
[ROW] 4: 446|NZR Addington|30-4-1910|6-12-1946|1910|1946""", question="how many number were in service in 1910?"),
"role": chat_api_user
},
{
"content": """SQL: ```SELECT COUNT(*) FROM DF;```.""",
"role": chat_api_assistant
},
{
"content": prompt_template.format(df="""[HEAD]: COUNT(*)
---
[ROW] 1: 4""", question="how many number were in service in 1910?"),
"role": chat_api_user
},
{
"content": """Answer: ```4```.""",
"role": chat_api_user
}
]

    print(f"Question over dataframe = {question}")
    print(f"input dataframe = {df}")
    
    # have the few_shot_examples at the top
    chat_history:List[Dict] = list(few_shot_examples)
    
    hops:int = 0
    while True:
        # serialize the df
        prompt_df_str = _serialize_df(df)
        
        # format the prompt and add to chat_history
        chat_history.append( { "content": prompt_template.format(df=prompt_df_str, question=question), "role": chat_api_user } )
        
        # execute the prompt
        gpt_output:dict = _gpt_api_caller(chat_history, engine, verbose)
        
        # output={
            #   "choices": [
            #     {
            #       "finish_reason": "stop",
            #       "index": 0,
            #       "logprobs": null,
            #       "message": {
            #         "content": "SQL: ```SELECT MAX(Year) FROM DF WHERE League='USL A-League';",
            #         "role": "assistant"
            #       },
            #       "text": "SQL: ```SELECT MAX(Year) FROM DF WHERE League='USL A-League';"
            #     }
            #   ],
            #   "created": 1710951470,
            #   "id": "chatcmpl-94svO7dcbERhzyM9H20cEcQw1DNvR",
            #   "model": "gpt-4-0613",
            #   "object": "chat.completion",
            #   "system_fingerprint": null,
            #   "usage": {
            #     "completion_tokens": 18,
            #     "prompt_tokens": 1405,
            #     "total_tokens": 1423
            #   }
            # }
            # output={
            #   "choices": [
            #     {
            #       "finish_reason": "stop",
            #       "index": 0,
            #       "logprobs": null,
            #       "message": {
            #         "content": "Python: ```DF[DF['port']=='Auckland'].sort_values(by='speed', ascending=False).iloc[0]['name']",
            #         "role": "assistant"
            #       },
            #       "text": "Python: ```DF[DF['port']=='Auckland'].sort_values(by='speed', ascending=False).iloc[0]['name']"
            #     }
            #   ],
            #   "created": 1710943509,
            #   "id": "chatcmpl-94qqzt0LZG2QRTBUm91FXS0vJdS1X",
            #   "model": "gpt-4-0613",
            #   "object": "chat.completion",
            #   "system_fingerprint": null,
            #   "usage": {
            #     "completion_tokens": 27,
            #     "prompt_tokens": 1794,
            #     "total_tokens": 1821
            #   }
            # }        
        # check if the output has sql, python or the answer
        gen_text:str = gpt_output["choices"][0]["text"]
        
        # SQL: ```SELECT MAX(Year) FROM DF WHERE League='USL A-League';
        # non greedy matching; ignore case
        sql_match:regex.Match = regex.match("^\s*sql:.*?```(.*)", gen_text, regex.IGNORECASE|regex.MULTILINE|regex.DOTALL)
        # Python: ```DF[DF['port']=='Auckland'].sort_values(by='speed', ascending=False).iloc[0]['name']
        # 
        # to accommodate this, don't use regex.match() which searches from start of the string:       gen_text = 'The speed extraction is incorrect. The speed in knots is the value before "knots" in the propulsion column, not the first value. Let\'s correct this:\n\nPython: ```\ndef get_speed(s):\n    return int(s.split(\' \')[-3])\nDF[\'speed\'] = DF.apply(lambda x: get_speed(x[\'propulsion\']), axis=1)\n'
        #
        # non greedy matching so that the start delimiter ``` is matched and not the end delimiter; ignore case; DOTALL == match newlines; MULTILINE == ^ and $ ignore new lines.
        python_match:regex.Match = regex.search("\s*python:.*?```(.*)", gen_text, regex.IGNORECASE|regex.MULTILINE|regex.DOTALL)
        if sql_match:
            print(f"About to execute SQL: {sql_match.group(1)}")
            df:pd.DataFrame = _exec_sql(df, sql_match.group(1))
            # append the generated text to chat_history.  Include the stop word ```. 
            chat_history.append( { "content": f"{gen_text}```.", "role":"assistant" } )
        elif python_match:
            print(f"About to execute Python: {python_match.group(1)}")
            _exec_python(df, python_match.group(1))
            # append the generated text to chat_history.  Include the stop word ```. 
            chat_history.append( { "content": f"{gen_text}```.", "role":"assistant" } )
        elif gen_text.lower().startswith("answer:"):
            print(f"Finished: Stopping: Answer: {gen_text}")
            chat_history.append( { "content": f"{gen_text}```.", "role":"assistant" } )
            return gen_text, chat_history
        else:
            raise Exception(f"Unknown text generation: gen_text={gen_text}; engine={engine}; chat_history={chat_history}")
        
        hops += 1
        if hops == max_hops: raise Exception(f"max_hops reached: max_hops{max_hops}; gen_text='{gen_text}'; engine={engine}; messages={chat_history}")

if __name__ == "__main__":
    # [HEAD]: Year|Division|League|Regular Season|Playoffs|Open Cup|Avg. Attendance
    # ---
    # [ROW] 1: 2001|2|USL A-League|4th, Western|Quarterfinals|Did not qualify|7,169
    # [ROW] 2: 2002|2|USL A-League|2nd, Pacific|1st Round|Did not qualify|6,260
    # [ROW] 3: 2003|2|USL A-League|3rd, Pacific|Did not qualify|Did not qualify|5,871
    # [ROW] 4: 2004|2|USL A-League|1st, Western|Quarterfinals|4th Round|5,628
    # [ROW] 5: 2005|2|USL First Division|5th|Quarterfinals|4th Round|6,028
    # [ROW] 6: 2006|2|USL First Division|11th|Did not qualify|3rd Round|5,575
    # [ROW] 7: 2007|2|USL First Division|2nd|Semifinals|2nd Round|6,851
    # [ROW] 8: 2008|2|USL First Division|11th|Did not qualify|1st Round|8,567
    # [ROW] 9: 2009|2|USL First Division|1st|Semifinals|3rd Round|9,734
    # [ROW] 10: 2010|2|USSF D-2 Pro League|3rd, USL (3rd)|Quarterfinals|3rd Round|10,727

    df_str = \
"""
Year|Division|League|Regular Season|Playoffs|Open Cup|Avg. Attendance
2001|2|USL A-League|4th, Western|Quarterfinals|Did not qualify|7,169
2002|2|USL A-League|2nd, Pacific|1st Round|Did not qualify|6,260
2003|2|USL A-League|3rd, Pacific|Did not qualify|Did not qualify|5,871
2004|2|USL A-League|1st, Western|Quarterfinals|4th Round|5,628
2005|2|USL First Division|5th|Quarterfinals|4th Round|6,028
2006|2|USL First Division|11th|Did not qualify|3rd Round|5,575
2007|2|USL First Division|2nd|Semifinals|2nd Round|6,851
2008|2|USL First Division|11th|Did not qualify|1st Round|8,567
2009|2|USL First Division|1st|Semifinals|3rd Round|9,734
2010|2|USSF D-2 Pro League|3rd, USL (3rd)|Quarterfinals|3rd Round|10,727
"""

    #df:pd.DataFrame = pd.read_csv(io.StringIO(df_str), sep='|')

    # question:str = "What was the last year where this team was a part of the usl a-league"
    # _iterate_prompt_exec_until_ans("gpt-4", df_str, '|', None, question)

    # question:str = "How many times did the team play in USL A-League?"
    # _iterate_prompt_exec_until_ans("gpt-4", df_str, '|', None, question)

    question:str = "Count the number of times the team played in each League?"
    (answer, messages) = iterate_prompt_exec_until_ans("gpt-4", df_str, '|', None, question)
    print(f"answer={answer}; messages={messages}")
    
