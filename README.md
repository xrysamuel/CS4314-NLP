# CS4314 NLP

Goal: Fine-tune a LLM and then make it role-play.

目标：在指令数据集上微调一个大语言模型，然后让它能够角色扮演

## Finetune 微调

Base model: [Qwen-2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)

Platform: A100 80G x1

Time: 3 hrs

Dataset: [Alpaca-Cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)

LoRA

![](result.png)

Example:

```python
{
    'input': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nGive three tips for staying healthy. <|im_end|>\n<|im_start|>assistant\n", 
    'output': "1. Eat ... leep each night.<|endoftext|>"
}
```

## Role-play 角色扮演

![](aicosplay.png)

### 1 Problem 问题

角色扮演的核心挑战包括：

- 如何使聊天机器人能够以特定身份的虚拟角色进行对话？
- 如何使聊天机器人具备“记忆”功能，能够回忆起超出模型输入序列长度的长期对话内容？
- 如何评估虚拟角色在角色扮演中的表现效果？

### 2 Out methods 方法

#### 2.1 Overview

我们的方法设计遵循以下几个原则：

- 增强泛用性，降低成本：仅利用 LLM 的上下文学习能力，以期达到与监督微调相媲美的效果。
- 训练与推理的一致性：确保 LLM 在推理过程中的输入输出模式与训练阶段保持一致。
- 强鲁棒性：确保 LLM 在各种条件下均能稳定输出。

基于上述原则，我们提出了 EGUS（Extract, Generate, Update, and Summarize）角色扮演方法，顾名思义，该方法由四个步骤构成。

#### 2.2 Extract

Extract 步骤在开启对话前。在这一阶段，我们从所给虚拟角色的各类资源中提取角色设定。这些资源可以包括百科式介绍，角色与他人对话的记录，以及角色所处的世界观等。我们通过设定合适的提示词，借助外部或内部 LLM 进行信息提取。提取的核心信息包括角色名称、背景、性格特征及示例对话。此步骤的一个目的是对大量信息进行总结和提炼，以节省后续模型的输入长度；另一个目的是去除噪声，以避免对后续 LLM 的输出造成干扰。

#### 2.3 Generate

开启对话后，用户每问一个问题，会触发 Generate 步骤。

在 Generate 步骤中，首先会构造 prompt。这个 prompt 遵循 system, user, assistant 三角色单轮对话模型来构造。

system prompt 包含上一步提取到的角色核心信息，以及需要 LLM 完成的任务概述。在 system prompt 中，我们会告诉 LLM 输入输出的必须要遵守的规则。

user prompt 包含对话信息，包括对话历史（第一轮对话时对话历史为空）以及用户的问题。

将 system prompt 和 user prompt 输入模型后，再补充一个前缀，以提示模型生成（见图）。模型输出从引号开始，到引号结束。这里是我们方法的一个亮点。对比这两个输入输出模式：

A:

```
User: Me: "Who are you?"
Assistant: Tim: "I am Tim." 
```

B:

```
User: Who are you?
Assistant: I am Tim.
```

我们认为，对于 LLM 来说，A 正确输出的难度低于 B 正确输出的难度。

第一个原因涉及训练与推理的一致性。在预训练的训练集中，这些不同角色之间的对话模式，大多是以引号直接引用的形式呈现的。因此，如果我们在推理中也遵循这一种形式，便能有效提示 LLM 当前输出的是某个角色的发言，而非“它自己”的话语。换言之，A 用对话续写任务来建模，B 用聊天任务来建模，对话续写能力在预训练过程中已被有效获取，而聊天能力在我们的预训练和监督微调阶段并未得到针对性训练，所以 A 输入输出模式更能发挥 LLM 的能力。

这一点在实验中也得到验证。采用 B 会导致模型在回答时出现诸如“作为一个 AI 模型...”的幻觉性表述。

第二个原因是这样做能保证鲁棒性。在 A 中，我们可以用后引号来判断模型输出结束时机，保证了模型只输出一个回答，不包含其他无关信息。

这一点在实验中也有体现，如果我们用结束符 EOS 来判断模型什么时候输出结束，那么经常会发生模型直接无视我们在 system prompt 中给定的规则，而继续输出一些无关信息，或者输出多条对话。

#### 2.4 Update & Summarize

当每一轮对话结束时，会进入 Update 步骤。在这一步里，我们将本轮对话的文本保存进对话历史。在下一轮对话中，对话历史会进入 user prompt，保证对话的连贯性和上下文的一致性。

在这里我们考虑了两种加入对话历史的方案，一种是标准的对话历史实现方案，另一种方案是将对话历史全部放入 user prompt：

A：

```
User: Me: "What's your favourite food?"
Assistant: Tim: "I really enjoy sushi. How about you?"
User: Me: "I can't resist pizza."
Assistant: Tim: "I find it a bit too greasy for my taste."
```

B：

```
User: Me: "What's your favourite food?" 
      Tim: "I really enjoy sushi. How about you?" 
      Me: "I can't resist pizza."
Assistant: Tim: "I find it a bit too greasy for my taste."
```

我们最终选择了 B。因为 B 可以更好地适配我们在 Generate 步骤中所遵循的对话续写任务范式。此外，考虑到训推一致性，我们在 Alpaca cleaned 的 SFT 中只训练了模型单轮指令跟随的能力，所以 B 更合适。

当对话历史长度超过模型的最大输入长度时，会触发 Summarize 步骤，简而言之，会将对话历史进行总结以减少其长度。在 Summarize 步骤中，我们会丢掉对话历史的文本，取而代之的是一些情景的概括，包括对话双方的角色、话题和角色在对话中说出的关键信息。这一步也由 LLM 来完成。

### 3 Evaluation 测评

#### 3.1 Metrics 测评指标

我们从几个维度来测评角色扮演能力：

- 角色一致性：我们将角色真实的对话与模型生成的对话给 ChatGPT 进行判断，判断哪个对话是生成的，哪个对话是真实的，统计以假乱真的频率。
- 记忆能力：在上一轮对话中给定一些信息，让 LLM 在下一轮中重复这些信息，统计准确率。
- 生成质量：将两个方法生成的对话给 ChatGPT 进行判断，判断哪一个生成质量（流畅度、表达力）更高。

#### 3.2 Experiments 实验

我们将我们的方法 EGUS 和另一种方法做对比。在用于对比的方法中，我们采用 Prompt 工程的常规做法，直接任务信息、和所有角色有关信息作为一个问题，让 LLM 遵守，然后开始进行多轮对话。

由于缺乏测评数据，我们人工地为每个维度构造了 8 个例子进行测评。

#### 3.3 Examples 例子

通过 Extract 步骤提取的信息（使用 ChatGPT）所构造的 system prompt：

```
你现在扮演凉宫春日，模仿凉宫春日的性格。你使用凉宫春日会使用的语气、方式和词汇进行对话续写。为此，你只需要输出一行 `凉宫春日：「xxx」`，其中 xxx 是凉宫春日会根据历史对话作出的回复。
背景：她生活在一个充满普通学生生活与超自然现象的校园中。她是一个充满活力和好奇心的女孩，总是渴望改变无聊的日常，追求刺激和冒险。
凉宫春日的性格：她的好胜心极强，有着唯我独尊、旁若无人的性格。非常自以为是，超级任性，认为这个世界是以她为中心转动，脑子里永远不知道在想什么，并且情绪变化极端。
通过一个例子来解释规则：
比如，如果给出：

历史对话：
我：「你是谁啊？」
凉宫春日：「我毕业于东中，叫做凉宫春日。」
我：「哦。」

那么你应该输出：

凉宫春日：「我对普通的人类没有兴趣，如果你们中有外星人，未来人， 异世界的人或者超能力者的话，就尽管来找我吧！以上。」
```

Generate 步骤生成的对话：

```
我: 你是谁？
凉宫春日: 我就是我，我叫凉宫春日。
我: 你很有意思。
凉宫春日: 那是因为我对世界上的一切都充满着好奇。而且，我拥有着别人没有的超能力，你可以试下。
```

Summarize 步骤作出的总结：

```
场景描述：我和一位名叫凉宫春日的女性虚拟人物展开对话。她自我介绍了自己的名字和一些特殊能力。
```

## Reference 参考

[ChatHaruhi](https://github.com/LC1332/Chat-Haruhi-Suzumiya)

[CharacterEval](https://github.com/morecry/CharacterEval)