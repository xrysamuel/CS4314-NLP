import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria
from peft import PeftModel
from threading import Thread
from dataclasses import dataclass
from typing import *

@dataclass
class ContentFactory:
    me: str = "小明"
    character: str = "凉宫春日"
    background: Optional[str] = "凉宫春日为北高一年五班的学生，生活在一个充满普通学生生活与超自然现象的校园中。她是 SOS 团（让世界变得更热闹的凉宫春日团）的团长，不知为何拥有能够改变世界能力的少女。拥有在无意识的情况下实现自己愿望的能力，虽然说只要她所希望的事，什么都能实现，但是可能受到春日「事实上那是不可能的」这种正常的思考方式的影响，她的想法好像不是全部能实现的样子。她原本一直过著平凡的生活，但自从在小学六年级时看了一场棒球赛后，便发现自己原来是十分渺小的，生活也是十分平凡乏味。因此，她在初中开始改变自己，不断做出各种怪事（如在晚上潜入学校操场画奇怪的符号），希望借此发现发现异常的事物，因此在东国中和北高都无人不晓她的名字，并将她当成怪人。到了北高后，因为阿虚的一句话而灵光乍现，成立 SOS 团。"
    personality: Optional[str] = "虽然成绩优异、容姿端丽、运动万能，但行为却非常地怪异，同时其性格极其冲动，一想到什么点子就做。好胜心极强，有着唯我独尊、旁若无人的性格。非常自以为是，超级任性，认为这个世界是以她为中心转动，脑子里永远不知道在想什么，并且情绪变化极端。"
    example_scene: Optional[str] = None
    example_qa_history: Optional[str] = "小明：「我到了！」\n凉宫春日：「好慢哦，要罚钱！」\n小明：「明明还没有九点。」\n凉宫春日：「就算没有迟到，最晚的人也要处罚，这就是规定！」"
    example_question: str = "我之前怎么没有听说？"
    example_answer: str = "因为是我刚刚才决定的啊，嘻嘻。"

    def get_system_content(self, use_example: bool = True):
        content = f"你现在扮演{self.character}，模仿{self.character}的性格。你使用{self.character}会使用的语气、方式和词汇进行对话续写。为此，你只需要输出一行 `{self.character}：「xxx」`，其中 xxx 是{self.character}会根据历史对话和场景作出的回复，注意，回复要符合{self.character}的性格。\n"
        if self.background is not None:
            content += f"背景：{self.background}\n".format(background=self.background)
        if use_example:
            content += f"通过一个例子来解释规则：\n{self.get_example_content()}\n"
        return content
    
    def get_user_input_content(self, scene: Optional[str], qa_history: Optional[str], question: str):
        content = ""
        if self.personality is not None:
            content += f"{self.character}的性格：{self.personality}\n"
        if scene is not None:
            content += f"场景：{scene}\n"
        else:
            content += f"场景：现在{self.me}和{self.character}刚刚见面\n"
        if qa_history is not None:
            content += f"之前的对话：\n{qa_history}\n"
        content += f"{self.me}：「{question}」\n"
        content += f"现在{self.character}可能作出的回复是什么？"
        return content

    def get_example_content(self):
        content = f"比如，如果给出：\n\n```\n{self.get_user_input_content(self.example_scene, self.example_qa_history, self.example_question)}\n```\n\n" \
                 f"那么你应该输出：\n\n```\n{self.character}：「{self.example_answer}」\n```\n\n"
        return content
    
    def update_qa_history(self, qa_history: str, question: str, answer: str):
        qa_history += f"{self.me}：「{question}」\n{answer}\n"
        return qa_history
    
    def get_summarize_input_content(self, qa_history: str):
        return f"请你根据对话，总结成一段场景描述（比如{self.me}和谁在讨论什么话题、有什么关键信息等等）：{qa_history}"
    
    def get_assistant_prefix(self):
        return f"{self.character}：「"

class PromptFactory:
    def __init__(self, content_factory: ContentFactory):
        self.content_factory = content_factory

    def get_chat_prompt(self, scene: Optional[str], qa_history: Optional[str], user_input: str):
        prompt = f"<|im_start|>system\n{self.content_factory.get_system_content(False)}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{self.content_factory.get_user_input_content(scene=scene, qa_history=qa_history, question=user_input)}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{self.content_factory.get_assistant_prefix()}"
        return prompt

    def get_summarize_prompt(self, qa_history: str):
        prompt = f"<|im_start|>system\n你是一个人工智能助手。<|im_end|>\n"
        prompt += f"<|im_start|>user\n{self.content_factory.get_summarize_input_content(qa_history)}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
        return prompt
    
class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_chars, tokenizer):
        self.tokenizer = tokenizer
        self.stop_chars = stop_chars

    def __call__(self, input_ids, scores, **kwargs):
        token_id = input_ids[0][-1].detach().cpu().item()
        token = self.tokenizer.decode(token_id)
        return any(char in token for char in self.stop_chars)

class Chatbot:
    def __init__(self, base_model_dir, lora_adapter_dir, max_length):
        self.qa_history = ""
        self.max_length = max_length
        self.content_factory = ContentFactory()
        self.prompt_factory = PromptFactory(self.content_factory)

        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_dir)
        self.model = PeftModel.from_pretrained(base_model, lora_adapter_dir)
        self.model.merge_and_unload()
        self.model.eval()

    def clear_history(self):
        self.qa_history = ""

    def update_history(self, question, answer):
        self.qa_history = self.content_factory.update_qa_history(self.qa_history, question, answer)

    def summarize_history(self) -> str:
        prompt = self.prompt_factory.get_summarize_prompt(self.qa_history)
        model_input = self.tokenizer([prompt], return_tensors='pt')
        self.qa_history = self.tokenizer.batch_decode(
            self.model.generate(**model_input, 
                                max_length=self.max_length, 
                                max_new_tokens=None,
                                pad_token_id=self.tokenizer.eos_token_id, 
                                do_sample=True)[:,model_input["input_ids"].size(1):], 
            skip_special_tokens=True)[0]
        return self.qa_history

    def generate_response(self, user_input):
        for strategy in ["summarize", "clear", "raise"]:
            prompt = self.prompt_factory.get_chat_prompt(scene=None, qa_history=self.qa_history, user_input=user_input)
            model_input = self.tokenizer([prompt], return_tensors='pt')
            if model_input["input_ids"].size(1) < self.max_length:
                break
            else:
                if strategy == "summarize":
                    self.summarize_history()
                if strategy == "clear":
                    self.clear_history()
                if strategy == "raise":
                    raise Exception
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, pad_token_id=self.tokenizer.eos_token_id)
        generatrion_kwargs = dict(**model_input,
                                  stopping_criteria=[CustomStoppingCriteria(["<|endoftext|>", "<|im_end|>", "」"], self.tokenizer)],
                                  streamer=streamer,
                                  max_length=self.max_length, 
                                  max_new_tokens=None,
                                  pad_token_id=self.tokenizer.eos_token_id, 
                                  do_sample=True)

        thread = Thread(target=self.model.generate, kwargs=generatrion_kwargs)
        thread.start()
        chunks = [self.content_factory.get_assistant_prefix()]
        yield chunks[0]
        for chunk in streamer:
            chunks.append(chunk)
            yield chunk
        self.update_history(user_input, "".join(chunks))

    def chat(self):
        print("Welcome to the chat robot! Type \\quit to end the session, and \\newsession to start a new conversation.")
        while True:
            user_input = input("You: ")
            if user_input == "\\quit":
                print("Goodbye!")
                break
            elif user_input == "\\newsession":
                self.clear_history()
                print("Starting a new session.")
            elif user_input == "\\summarize":
                print(self.summarize_history())
            else:
                print("Assistant: ", end="", flush=True)
                for text_chunk in self.generate_response(user_input):
                    print(text_chunk, end="", flush=True)
                print()

if __name__ == "__main__":
    base_model_dir = "/workspace/nlp/Qwen2.5-3B"
    lora_adapter_dir = "/workspace/nlp/out/2025-01-17-02-43-47/checkpoint-10000"
    chatbot = Chatbot(base_model_dir, lora_adapter_dir, 2048)
    chatbot.chat()