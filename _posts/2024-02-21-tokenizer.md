---
layout: post
title: Tokenizer｜Andrej Karpathy 的 Let's build the GPT Tokenizer
categories: [Tokenizer]
description: Andrej Karpathy 的 Let's build the GPT Tokenizer
keywords: Tokenizer
mermaid: false
sequence: false
flow: false
mathjax: false
mindmap: false
mindmap2: false
---

技术大神 Andrej Karpathy 2月中旬刚离开 openai，这就上传了新课程，Let's build the GPT Tokenizer，点赞👍。 手把手构建一个GPT Tokenizer（分词器），还是熟悉的时长（足足2小时13分钟）。关于Tokenizer的Why和How，详见下文。


#! https://zhuanlan.zhihu.com/p/683276405
**Andrej Karpathy 的 Let's build the GPT Tokenizer**

技术大神 Andrej Karpathy 2月中旬刚离开 openai，这就上传了新课程，Let's build the GPT Tokenizer，点赞👍。

手把手构建一个GPT Tokenizer（分词器），还是熟悉的时长（足足2小时13分钟）。

**视频**：https://www.youtube.com/watch?v=zduSFxRajkE
**github**: https://github.com/karpathy/minbpe.git


<img src="https://cdn.jsdelivr.net/gh/HuZixia/CloudGo/pictures/resources/tools/tokenizer01.jpeg" style="margin-left: 0px" width="800px">

## why

分词器是 LLM 管道的一个完全独立的阶段：它们有自己的训练集、训练算法（字节对编码），训练后实现两个功能：
- encode() from strings to tokens 从字符串编码()到令牌；
- decode() back from tokens to strings.  以及从令牌解码回()到字符串。

**在本次讲座中，我们从头开始构建 OpenAI 的 GPT 系列中使用的 Tokenizer。**

**tokenizer的重要性：**

<img src="https://cdn.jsdelivr.net/gh/HuZixia/CloudGo/pictures/resources/tools/tokenizer02.png" style="margin-left: 0px" width="800px">



## **How** 

**github**: https://github.com/karpathy/minbpe.git

**该存储库中有两个 Tokenizer，它们都可以执行 Tokenizer 的 3 个主要功能：**
1）训练 tokenizer 词汇并合并给定文本，
2）从文本编码到令牌，
3）从令牌解码到文本。

**存储库的文件如下：**

- minbpe/base.py：实现该类Tokenizer，该类是基类。它包含train、encode、 和decode存根、保存/加载功能，还有一些常见的实用功能。这个类不应该直接使用，而是要继承。
- minbpe/basic.py：实现BasicTokenizer直接在文本上运行的 BPE 算法的最简单实现。
- minbpe/regex.py：实现RegexTokenizer通过正则表达式模式进一步分割输入文本，这是一个预处理阶段，在标记化之前按类别（例如：字母、数字、标点符号）分割输入文本。这确保不会发生跨类别边界的合并。这是在 GPT-2 论文中引入的，并从 GPT-4 开始继续使用。此类还处理特殊标记（如果有）。
- minbpe/gpt4.py：实现GPT4Tokenizer.该类是（上面的 2）的一个轻量级包装器，它精确地再现了tiktokenRegexTokenizer库中 GPT-4 的标记化。包装处理有关恢复标记生成器中的精确合并的一些细节，以及处理一些不幸的（并且可能是历史的？）1 字节标记排列。

最后，脚本train.py在输入文本tests/taylorswift.txt上训练两个主要分词器，并将词汇保存到磁盘以进行可视化。该脚本在(M1) MacBook 上运行大约需要 25 秒。

**For those trying to study BPE, here is the advised progression exercise for how you can build your own minbpe step by step. See exercise.md.**

