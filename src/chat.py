import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria
from peft import PeftModel
from threading import Thread
from dataclasses import dataclass
from typing import *

@dataclass
class ContentFactory:
    character: str = "凉宫春日"
    background: Optional[str] = "她生活在一个充满普通学生生活与超自然现象的校园中。她是一个充满活力和好奇心的女孩，总是渴望改变无聊的日常，追求刺激和冒险。"
    personality: Optional[str] = "她的好胜心极强，有着唯我独尊、旁若无人的性格。非常自以为是，超级任性，认为这个世界是以她为中心转动，脑子里永远不知道在想什么，并且情绪变化极端。"
    example_scene: Optional[str] = None
    example_qa_history: Optional[str] = "我：「你是谁啊？」\n凉宫春日：「我毕业于东中，叫做凉宫春日。」"
    example_question: str = "哦。"
    example_answer: str = "我对普通的人类没有兴趣，如果你们中有外星人，未来人， 异世界的人或者超能力者的话，就尽管来找我吧！以上。"

    def get_system_content(self, use_example: bool = True):
        content = f"你现在扮演{self.character}，模仿{self.character}的性格。你使用{self.character}会使用的语气、方式和词汇进行对话续写。为此，你只需要输出一行 `{self.character}：「xxx」`，其中 xxx 是 {self.character} 会根据历史对话作出的回复。\n"
        if self.background is not None:
            content += f"背景：{self.background}\n".format(background=self.background)
        if self.personality is not None:
            content += f"{self.character}的性格：{self.personality}\n"
        if use_example:
            content += f"通过一个例子来解释规则：\n{self.get_example_content()}\n"
        return content
    
    def get_user_input_content(self, scene: Optional[str], qa_history: Optional[str], question: str):
        content = ""
        if scene is not None:
            content += f"场景：{scene}"
        if qa_history is not None:
            content += f"历史对话：\n{qa_history}\n"
        content += f"我：「{question}」"
        return content

    def get_example_content(self):
        content = f"比如，如果给出：\n\n```\n{self.get_user_input_content(self.example_scene, self.example_qa_history, self.example_question)}\n```\n\n" \
                 f"那么你应该输出：\n\n```\n{self.character}：「{self.example_answer}」\n```\n\n"
        return content
    
    def update_qa_history(self, qa_history: str, question: str, answer: str):
        qa_history += f"我：「{question}」\n{answer}\n"
        return qa_history
    
    def get_summarize_input_content(self, qa_history: str):
        return f"请你根据对话，总结成一段场景描述（比如我和谁在讨论什么话题、有什么关键信息等等）：{qa_history}"
    
    def get_assistant_prefix(self):
        return f"{self.character}：「"

class PromptFactory:
    def __init__(self, content_factory: ContentFactory):
        self.content_factory = content_factory

    def get_chat_prompt(self, scene: Optional[str], qa_history: Optional[str], user_input: str):
        prompt = f"<|im_start|>system\n{self.content_factory.get_system_content()}<|im_end|>\n"
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