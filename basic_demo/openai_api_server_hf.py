import time
import sys
import re
import random
import string

from asyncio.log import logger

from loguru import logger

import uvicorn
import gc
import json
import torch

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union, Tuple
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, LogitsProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from sse_starlette.sse import EventSourceResponse

from aiostream.stream import list as alist


logger.remove()  # 这行很关键，先删除logger自动产生的handler，不然会出现重复输出的问题
#logger.add(sys.stderr, level='TRACE')  # 只输出警告以上的日志
#logger.add(sys.stderr, level='WARNING')  # 只输出警告以上的日志
logger.add(sys.stderr, level='INFO')  # 只输出警告以上的日志


EventSourceResponse.DEFAULT_PING_INTERVAL = 10000

MODEL_PATH = 'THUDM/glm-4-9b-chat'
MAX_MODEL_LENGTH = 8192


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_id(prefix: str, k=29) -> str:
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
    return f"{prefix}{suffix}"


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = ["glm-4"]


class FunctionCall(BaseModel):
    name: str
    arguments: str


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionMessageToolCall(BaseModel):
    index: Optional[int] = 0
    id: Optional[str] = None
    function: FunctionCall
    type: Optional[Literal["function"]] = 'function'


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]
    index: int


class ChatCompletionResponse(BaseModel):
    model: str
    id: Optional[str] = Field(default_factory=lambda: generate_id('chatcmpl-', 29))
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    system_fingerprint: Optional[str] = Field(default_factory=lambda: generate_id('fp_', 9))
    usage: Optional[UsageInfo] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    tool_choice: Optional[Union[str, dict]] = "None"
    repetition_penalty: Optional[float] = 1.1


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def process_response(output: str, use_tool: bool = False) -> Union[str, dict]:
    lines = output.strip().split("\n")
    arguments_json = None
    special_tools = ["cogview", "simple_browser"]

    tool_call_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    if len(lines) >= 2 and tool_call_pattern.match(lines[0]):
        function_name = lines[0].strip()
        arguments = "\n".join(lines[1:]).strip()

        try:
            arguments_json = json.loads(arguments)
            is_tool_call = True
        except json.JSONDecodeError:
            is_tool_call = function_name in special_tools

        if is_tool_call and use_tool:
            content = {
                "name": function_name,
                "arguments": json.dumps(arguments_json if isinstance(arguments_json, dict) else arguments, ensure_ascii=False)
            }
            if function_name == "simple_browser":
                search_pattern = re.compile(r'search\("(.+?)"\s*,\s*recency_days\s*=\s*(\d+)\)')
                match = search_pattern.match(arguments)
                if match:
                    content["arguments"] = json.dumps({
                        "query": match.group(1),
                        "recency_days": int(match.group(2))
                    }, ensure_ascii=False)
            elif function_name == "cogview":
                content["arguments"] = json.dumps({
                    "prompt": arguments
                }, ensure_ascii=False)

            return content
    return output.strip()

def apply_stopping_strings(reply, stop_strings) -> Tuple[str, bool]:
    stop_found = False
    for string in stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou: is completed, trim it
        for string in stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue
            break
    return reply, stop_found



@torch.inference_mode()
def generate_stream_glm4(params: dict):
    global engine, tokenizer

    echo = params.get("echo", True)
    messages = params["messages"]
    tools = params["tools"]
    tool_choice = params["tool_choice"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 8192))
    messages = process_messages(messages, tools=tools, tool_choice=tool_choice)
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt",return_dict=True)

    inputs = inputs.to(engine.device)
    input_echo_len = len(inputs["input_ids"][0])

    if input_echo_len >= engine.config.seq_length:
        print(f"Input length larger than {model.config.seq_length}")

    eos_token_id = [tokenizer.eos_token_id, 
        tokenizer.convert_tokens_to_ids("<|user|>"),
        tokenizer.convert_tokens_to_ids("<|observation|>")]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len = 0
    for total_ids in engine.stream_generate(**inputs, eos_token_id=eos_token_id, **gen_kwargs):
        total_ids = total_ids.tolist()[0]
        total_len = len(total_ids)

        if echo:
            output_ids = total_ids[:-1]
        else:
            output_ids = total_ids[input_echo_len:-1]

        response = tokenizer.decode(output_ids)

        logger.debug(f"==== response ====\n{response}")

        if response and response[-1] != "�":
        
            logger.debug(f"==== apply_stopping_strings 1 ====\n{response}")
            response, stop_found = apply_stopping_strings(response, ["<|observation|>"])
            logger.debug(f"==== apply_stopping_strings 2 ====\n{response}")
            logger.debug(f"==== apply_stopping_strings：stop_found ====\n{stop_found}")

            yield {
                "text": response,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": total_len - input_echo_len,
                    "total_tokens": total_len,
                },
                "finish_reason": "function_call" if stop_found else None,
            }

            if stop_found:
                break

    # Only last stream result contains finish_reason, we set finish_reason as stop
    ret = {
        "text": response,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
        "finish_reason": "stop",
    }
    yield ret

    gc.collect()
    torch.cuda.empty_cache()



def process_messages(messages, tools=None, tool_choice="none"):
    _messages = messages
    processed_messages = []
    msg_has_sys = False

    def filter_tools(tool_choice, tools):
        function_name = tool_choice.get('function', {}).get('name', None)
        if not function_name:
            return []
        filtered_tools = [
            tool for tool in tools
            if tool.get('function', {}).get('name') == function_name
        ]
        return filtered_tools

    if tool_choice != "none":
        if isinstance(tool_choice, dict):
            tools = filter_tools(tool_choice, tools)
        if tools:
            processed_messages.append(
                {
                    "role": "system",
                    "content": None,
                    "tools": tools
                }
            )
            msg_has_sys = True

    if isinstance(tool_choice, dict) and tools:
        processed_messages.append(
            {
                "role": "assistant",
                "metadata": tool_choice["function"]["name"],
                "content": ""
            }
        )

    for m in _messages:
        role, content, func_call = m.role, m.content, m.function_call
        tool_calls = getattr(m, 'tool_calls', None)

        if role == "function":
            processed_messages.append(
                {
                    "role": "observation",
                    "content": content
                }
            )
        elif role == "tool":
            processed_messages.append(
                {
                    "role": "observation",
                    "content": content,
                    "function_call": True
                }
            )
        elif role == "assistant":
            if tool_calls:
                for tool_call in tool_calls:
                    processed_messages.append(
                        {
                            "role": "assistant",
                            "metadata": tool_call.function.name,
                            "content": tool_call.function.arguments
                        }
                    )
            else:
                for response in content.split("\n"):
                    if "\n" in response:
                        metadata, sub_content = response.split("\n", maxsplit=1)
                    else:
                        metadata, sub_content = "", response
                    processed_messages.append(
                        {
                            "role": role,
                            "metadata": metadata,
                            "content": sub_content.strip()
                        }
                    )
        else:
            if role == "system" and msg_has_sys:
                msg_has_sys = False
                continue
            processed_messages.append({"role": role, "content": content})

    if not tools or tool_choice == "none":
        for m in _messages:
            if m.role == 'system':
                processed_messages.insert(0, {"role": m.role, "content": m.content})
                break
    return processed_messages



@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="glm-4")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
        tool_choice=request.tool_choice,
    )
    logger.debug(f"==== request ====\n{gen_params}")

    if request.stream:
        predict_stream_generator = predict_stream(request.model, gen_params)
        return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")
    
        # output = next(predict_stream_generator)
        # if output:
        #     return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")
        
        
        # logger.debug(f"First result output：\n{output}")

        # function_call = None
        # if output and request.tools:
        #    try:
        #        function_call = process_response(output, use_tool=True)
        #    except:
        #        logger.warning("Failed to parse tool call")

        # # CallFunction
        # if isinstance(function_call, dict):
        #    function_call = FunctionCallResponse(**function_call)
        #    tool_response = ""
        #    if not gen_params.get("messages"):
        #        gen_params["messages"] = []
        #    gen_params["messages"].append(ChatMessage(role="assistant", content=output))
        #    gen_params["messages"].append(ChatMessage(role="tool", name=function_call.name, content=tool_response))
        #    generate = predict(request.model, gen_params)
        #    return EventSourceResponse(generate, media_type="text/event-stream")
        # else:
        #    generate = parse_output_text(request.model, output)
        #    return EventSourceResponse(generate, media_type="text/event-stream")


    response = ""
    for response in generate_stream_glm4(gen_params):
        pass

    if response["text"].startswith("\n"):
        response["text"] = response["text"][1:]
    response["text"] = response["text"].strip()

    usage = UsageInfo()

    function_call, finish_reason = None, "stop"
    tool_calls = None
    if request.tools:
        try:
            function_call = process_response(response["text"], use_tool=True)
        except Exception as e:
            logger.warning(f"Failed to parse tool call: {e}")

    if isinstance(function_call, dict):
        finish_reason = "tool_calls"
        function_call_response = FunctionCallResponse(**function_call)
        function_call_instance = FunctionCall(
            name=function_call_response.name,
            arguments=function_call_response.arguments
        )
        tool_calls = [
            ChatCompletionMessageToolCall(
                id=f"call_{int(time.time() * 1000)}",
                function=function_call_instance,
                type="function")]

    message = ChatMessage(
        role="assistant",
        content=None if tool_calls else response["text"],
        function_call=None,
        tool_calls=tool_calls,
    )

    logger.debug(f"==== message ====\n{message}")

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(
        model=request.model,
        id="",  # for open_source model, id is empty
        choices=[choice_data],
        object="chat.completion",
        usage=usage
    )


async def predict(model_id: str, params: dict):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, id="", choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    async for new_response in generate_stream_glm4(params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode

        finish_reason = new_response["finish_reason"]
        if len(delta_text) == 0 and finish_reason != "tool_calls":
            continue

        function_call = None
        if finish_reason == "tool_calls":
            try:
                function_call = process_response(decoded_unicode, use_tool=True)
            except:
                logger.warning(
                    "Failed to parse tool call, maybe the response is not a tool call or have been answered.")

        if isinstance(function_call, dict):
            function_call = FunctionCallResponse(**function_call)

        delta = DeltaMessage(
            content=None,
            role="assistant",
            function_call=None,
            tool_calls=[{
                "id": f"call_{int(time.time() * 1000)}",
                "index": 0,
                "type": "function",
                "function": function_call
            }] if isinstance(function_call, FunctionCallResponse) else None,
        )

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            id="",
            choices=[choice_data],
            object="chat.completion.chunk"
        )
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        id="",
        choices=[choice_data],
        object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


def predict_stream(model_id, gen_params):
    output = ""
    is_function_call = False
    has_send_first_chunk = False
    function_name = None
    created_time = int(time.time())
    response_id = generate_id('chatcmpl-', 29)
    system_fingerprint = generate_id('fp_', 9)
    
    
    logger.debug(f"==== response:gen_params ====\n{gen_params}")
    
    #async for new_response in generate_stream_glm4(gen_params):
    for new_response in generate_stream_glm4(gen_params):
        logger.debug(f"==== gen_params ====\n{gen_params}")
        logger.debug(f"==== new_response ====\n{new_response}")
        
        logger.debug(f"==== response:finish_reason ====\n{new_response['finish_reason']}")
        # finish_reason
        
        # 函数调用
        if new_response['finish_reason']:
            fc_tools = []
            if gen_params.get('tools'):
                for i in gen_params.get('tools'):
                    fc_tools.append(i.get('function').get('name'))
            if new_response["text"].split('\n')[0].strip() in fc_tools:
                # 需要函数调用
                is_function_call = True

                logger.info(f"====  函数调用   ====")
                function_name = new_response["text"].split('\n')[0].strip()
                args = eval("".join(new_response["text"].split('\n')[1:]).strip())
                args = json.dumps(args, ensure_ascii=False)
                
                logger.info(f"==== tool_function  ==== \n{function_name}")
                logger.info(f"==== tool_function_args  ==== \n{args}")

                for char in args:
                    function_call = {"name": function_name, "arguments": char}
                    tool_call = ChatCompletionMessageToolCall(
                        index=0,
                        function=FunctionCall(**function_call),
                        type="function"
                    )
                    message = DeltaMessage(
                        content=None,
                        role=None,
                        function_call=None,
                        tool_calls=[tool_call]
                    )
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=message,
                        finish_reason=None
                    )
                    chunk = ChatCompletionResponse(
                        model=model_id,
                        id=response_id,
                        choices=[choice_data],
                        created=created_time,
                        system_fingerprint=system_fingerprint,
                        object="chat.completion.chunk"
                    )
                    yield chunk.model_dump_json(exclude_unset=True)


                yield ChatCompletionResponse(
                                                model=model_id,
                                                id=response_id,
                                                system_fingerprint=system_fingerprint,
                                                choices=[
                                                    ChatCompletionResponseStreamChoice(
                                                        index=0,
                                                        delta=DeltaMessage(
                                                            content=None,
                                                            role=None,
                                                            function_call=None,
                                                        ),
                                                        finish_reason="tool_calls"
                                                    )],
                                                created=created_time,
                                                object="chat.completion.chunk",
                                                usage=None
                                            ).model_dump_json(exclude_unset=True)
                                                    
                
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(output):]
        output = decoded_unicode
        lines = output.strip().split("\n")
        
        


        if not is_function_call and len(output) > 7:
            logger.info(f'==== new_response["text"]  ==== \n{decoded_unicode}')
        
            finish_reason = new_response["finish_reason"]
            send_msgs = delta_text if has_send_first_chunk else output
            logger.debug(f"==== send_msg ====\n{send_msgs}")
            logger.debug(f"==== finish_reason ====\n{finish_reason}")
            

            has_send_first_chunk = True
            
            delta = DeltaMessage(
                role="assistant",
                content=send_msgs,
                function_call=None
            )
            
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
            )
                    
            chunk = ChatCompletionResponse(
                                            model=model_id,
                                            id="",  # for open_source model, id is empty
                                            choices=[choice_data],
                                            object="chat.completion",
                                            )
            logger.debug(f"==== chunk ====\n{chunk.model_dump_json(exclude_unset=True)}")
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    yield '[DONE]'
        



async def parse_output_text(model_id: str, value: str):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant", content=value),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, id="", choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, id="", choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


if __name__ == "__main__":
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    engine = AutoModelForCausalLM.from_pretrained(MODEL_PATH,trust_remote_code=True,quantization_config=BitsAndBytesConfig(load_in_4bit=True),torch_dtype=torch.bfloat16,device_map="cuda",).eval()
    
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
