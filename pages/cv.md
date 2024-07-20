---
layout: cv
title: 个人简介
description: 个人简介
keywords: CV, 推荐算法, 大模型微调
comments: true
menu: CV
permalink: /cv/
mermaid: false
sequence: false
flow: false
mathjax: false
mindmap: false
mindmap2: false
---


<div align="center">

:woman_technologist: <strong>Hi 👋 there, I'm</strong> <strong><a href="https://huzixia.github.io/">huzixia</a></strong> <img height="30" src="../images/work.gif" />

<div>&nbsp;</div>

  <!-- dynamic typing effect 动态打字效果 -->
  <div>
    <a href="https://huzixia.github.io/">
      <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&width=435&lines=console.log(%22Hello%2C%20World%22);胡同学祝您心想事成!&center=true&size=27" />
    </a>
  </div>
  <!-- profile logo 个人资料徽标 -->
  <div>
    <a href="https://huzixia.github.io/"><img src="https://img.shields.io/badge/Website-博客-orange" /></a>&emsp;
    <a href="https://www.zhihu.com/people/hu-zi-xia-91"><img src="https://img.shields.io/badge/ZhiHu-知乎-blue" /></a>&emsp;
    <a href="https://github.com/HuZixia"><img src="https://img.shields.io/badge/GitHub-code-white" /></a>&emsp;
    <a href="https://twitter.com/zixia80631/"><img src="https://img.shields.io/badge/Twitter-推特-black" /></a>&emsp;
    <a href="https://github.com/HuZixia/Text2Video/assets/38995480/244e64be-3dc4-46bb-8aff-523d8a235a1e"><img src="https://img.shields.io/badge/WeChat-微信-07c160" /></a>&emsp;

  </div>

</div>


## **求职意向**

<div style="display: flex; justify-content: space-between;">
  <div style="width: 48%; line-height: 1.2; margin: 0;">
    <p><strong>岗位：</strong>算法工程师</p>
  </div>
  <div style="width: 48%; line-height: 1.2; margin: 0;">
    <p><strong>坐标：</strong>北京</p>
  </div>
</div>


## **基本信息**

<div style="display: flex; justify-content: space-between;">
  <div style="width: 48%; line-height: 1.2; margin: 0;">
    <p><strong>姓名：</strong>胡紫霞</p>
    <p><strong>学校：</strong>中国农业大学</p>
    <p><strong>学历：</strong>985 硕士</p>
    <p><strong>英语：</strong>GRE, CET-6</p>
    <p><strong>工龄：</strong>8年</p>
  </div>
  <div style="width: 48%; line-height: 1.2; margin: 0;">
    <p><strong>电话：</strong>17600431688</p>
    <p><strong>微信：</strong>17600431688</p>
    <p><strong>邮箱：</strong>zixiahu2024@gmail.com</p>
    <p><strong>代码：</strong>MetaGPT 代码贡献者</p>
    <p><strong>博客：</strong>https://huzixia.github.io</p>
  </div>
</div>


## **专业证书**

<div style="display: flex; justify-content: space-between;">
  <div style="width: 48%; line-height: 1.2; margin: 0;">
    <p><strong>工信部：</strong>人工智能—大模型与AIGC 高级证书</p>
    <p><strong>工信部：</strong>大数据分析师 高级证书</p>
  </div>
  <div style="width: 48%; line-height: 1.2; margin: 0;">
    <p><strong>Agent：</strong>AI Agent Developer 证书</p>
    <p><strong>Microsoft：</strong>AI Applied Skills 证书</p>
  </div>
</div>


---

<div style="display: flex; justify-content: space-between;">
  <div style="line-height: 1.2; margin: 0;">
    <p><strong>AI创客松：</strong>Multi-Agent for X 最佳人气奖 （阿里云 & 魔搭社区 & Datawhale）</p>
    <p><strong>开源社区：</strong>动手学 AI 视频生成 最佳视频奖 （Datawhale & 奇想星球）</p>
    <p><strong>企业荣誉：</strong>多次 荣获企业年度 A级 绩效奖、24薪等</p>
    <p><strong>学校荣誉：</strong>多次 荣获国家奖学金、国家励志奖学金等</p>
  </div>
</div>


[//]: # (&emsp;)

## **专业技能**


{% for cv in site.data.cv %}
#### {{ cv.name }}
<div class="btn-inline">
{% for info in cv.keywords %}
<button class="btn btn-outline" type="button">{{ info }}</button>
{% endfor %}
</div>
{% endfor %}

---
[//]: # (&emsp;)


## **工作经历**


<div style="display: flex; justify-content: space-between;">
    <p style="font-size: 20px;"><strong>算法工程师 — 北京海纳金川科技有限公司</strong></p> 
    <p style="font-size: 20px;"><strong>2023 — 2024</strong></p>
</div>

在科技平台工作期间，负责将大模型应用到实际项目中，包括大模型微调、智能客服、视频生成以及自然专家系统等。这些项目旨在提升智能化水平、内容生成能力和用户体验等。

#### 项目一：模型微调

- **项目技术：** PEFT + ChatGLM + LLaMA + SFT + Lora + P-Tuning V2 + QLora + DeepSpeed 
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **项目简介：** 为了提升智能客服系统的响应效率和准确性，本项目针对用户数据进行大模型微调。通过应用LoRA、P-Tuning V2和QLora等微调技术，对ChatGLM和LLaMA模型进行优化，以更好地理解用户需求并提供精准回复。采用BLEU、准确率、召回率、F1值等指标评估模型微调的效果。并结合DeepSpeed进行性能优化，实现模型的高效训练和资源利用，最终提升用户满意度。
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **职责描述：**
  - **数据处理：** 收集和整理历史对话数据，对数据进行清洗和标注，为模型训练提供高质量的数据，构建模型微调格式的数据集。
  - **模型微调：** 开发优化模型代码，适配LoRA（降低finetune参数量）、P-Tuning V2和QLora（降低模型参数量）等微调技术，调整模型超参，对ChatGLM和LLaMA模型进行微调训练优化。集成DeepSpeed技术，实现模型的分布式训练，尝试ZeRO-1/2/3，通过分解optimizer states, gradients and weights，显著减少显存占用，使得在有限资源下训练更大规模模型成为可能。
  - **模型评估：** 评估模型微调的效果，针对Function Calling的每个参数（即Slot），评估准确率、召回率、F1值；针对文本回复，评估输出文本与参考文本之间的BLEU Score。

---

#### 项目二：智能客服

- **项目技术：** RAG + LangChain + Function Calling + Text Embedding + Chromadb + Hybrid Search + ChatGLM + GPT-4V
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **项目简介：** 为了提高用户体验和服务效率，本项目设计开发了一个基于大语言模型的智能客服系统。希望能够准确理解用户问题，并提供相关的解决方案和建议。通过构建一个包含常见问题答案、操作指南和服务流程的知识库，利用RAG技术，从知识库中检索相关信息，结合Hybrid Search方法提高检索效率和准确性，提高大模型回复的质量。系统还具备处理表格信息的能力。
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **职责描述：**
  - **系统设计：** 设计智能客服系统的整体架构，确保系统能够准确响应用户问题。包括选择和配置合适的大模型、设计系统模块和数据流，以及集成各种功能组件。
  - **知识库构建：** 收集、整理和存储知识信息，包括常见问题答案、操作指南和服务流程等。确保知识库信息准确性和及时更新。
  - **RAG集成：** 实现RAG技术，根据用户输入，通过关键字检索、向量检索模型和排序模型，从知识库获取参考答案。将用户输入和参考答案一起提供给大语言模型，生成最终回复。
  - **表格处理：** 处理PDF文档中的表格信息，通过OCR技术提取表格信息，或利用GPT-4V生成表格描述，然后向量化用于检索。或采用GPT-4V API做表格问答等。

[//]: # (向量检索模型)
[//]: # (封装 OpenAI 的 Embedding 模型接口 text-embedding-ada-002)

[//]: # (排序模型，信息检索重排序模型)
[//]: # (from sentence_transformers import CrossEncoder)
[//]: # (model = CrossEncoder&#40;'cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512&#41; # 英文，模型较小)
[//]: # (model = CrossEncoder&#40;'BAAI/bge-reranker-large', max_length=512&#41; # 多语言，国产，模型较大)

[//]: # (PDF文档中的表格如何处理：)
[//]: # (1. 表格结构，CSV，Embedding)
[//]: # (2. OCR，文字，Embedding)
[//]: # (3. GPT-4V，描述，Embedding)
[//]: # (4. 表格标题，Embedding)
[//]: # (5. 多模态模型，CLIP/BLIP)

[//]: # (微软开发的一个基于 Transformer 的模型，用于表格检测任务。该模型的主要作用是从图像中检测和识别表格结构。)
[//]: # (from transformers import AutoModelForObjectDetection)
[//]: # (# 加载 TableTransformer 模型)
[//]: # (model = AutoModelForObjectDetection.from_pretrained&#40;"microsoft/table-transformer-detection"&#41;)


---

#### 项目三：视频生成

- **项目技术：** Agent + Stable Diffusion + DALL-E + 通义万象 + TTS + Prompt + Moviepy + ChatGLM
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **项目简介：** 本项目旨在开发一个智能视频生成系统，通过整合多种先进技术，实现从文本、图片和视频生成高质量视频内容。系统基于多模态大模型，结合Agent协作机制，支持从文本生成视频、从文字和图片生成视频，以及从文字和现有视频生成新的视频。项目利用SD、DALL-E、通义万象生成图像，使用TTS技术生成语音，采用Moviepy合成视频。根据用户反馈迭代优化，提升效果。
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **职责描述：**
  - **系统开发：** 设计和开发视频生成系统，涵盖文字模块、语音模块、音乐模块、图像模块、视频模块等。集成各个模块，确保功能齐全，能够高效、准确地生成符合用户需求的视频内容。
  - **Agent协作：** 实现多个Agent的协作，确保不同模块的无缝衔接和高效运行，实现workflow多模态大模型协作，减少延迟和错误。
  - **功能模块：** 支持三种功能，文本生成视频，根据输入的文本生成剧本，进行分镜，然后生成图片，结合语音，合成视频；从文字和图片生成视频；从文字和现有视频生成新的视频。
  - **前后一致：** 在Prompt上进行微调，添加记忆功能，增强前后图片的角色一致性和内容连贯性，优化视频效果，提高用户满意度。


---

<div style="display: flex; justify-content: space-between;">
    <p style="font-size: 20px;"><strong>算法工程师 — 北京华品博睿网络技术有限公司</strong></p> 
    <p style="font-size: 20px;"><strong>2018 — 2023</strong></p>
</div>

在招聘平台工作期间，负责三大业务线：**BC端的店长直聘、BOSS直聘、蓝领交付** 。这些业务线的核心目标是通过推荐系统提升用户的招聘和求职体验。主要任务是设计和优化三大业务线的推荐策略和算法，包括召回、粗排、精排、重排等各个环节。此外，还负责模型校验、样本体系、特征工程、数据挖掘、商业增长等方面。



#### 项目一：重排算法

- **项目技术：** MTL + PSO + CEM + PointWise + PairWise + 级联模型 + 融合模型 + (1+1)-ES + RL + DDPG
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **项目简介：** 重排算法项目旨在提升推荐系统的排序性能，通过多目标融合的全局寻优和个性化寻优等技术手段，提高推荐结果的精确性和个性化。项目使用多任务学习MTL技术，将查看、开聊、回复、达成、拒绝等多种目标进行融合优化。采用PSO和CEM进行全局寻优，通过级联模型和融合模型进行个性化寻优，结合进化策略和强化学习方法，进一步优化推荐效果，满足用户多样化需求。
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **职责描述：**
  - **算法设计：** 负责重排算法的设计与实现，将查看、开聊、回复、达成、拒绝等多种目标进行融合优化，提高推荐结果的精确性和个性化，优化推荐效果。
  - **全局寻优：** 全局离线寻优，采用粒子群优化算法PSO，根据离线指标的约束，寻找多目标的最优组合。全局在线寻优，用CEM方法，只需定义策略参数和reward，直接优化线上转化目标。
  - **个性化寻优：** 结合user、user-item等维度特征，用模型学习个性化需求，融合多目标得分。具体方式有：
    - **级联模型：** PointWise和PairWise两种形式的级联模型(PLE+DeepFM)，主次模型解耦，精排层+重排层的级联模型。 
    - **融合模型：** 人工融合和帕累托融合两种形式的融合模型，端到端地通过一个辅助网络学习如何组合不同的目标，模型融合网络输出多目标融合参数。
    - **强化学习：** 构建四元组（状态S_t、动作A_t、收益Reward_t、状态S_t+1），训练DDPG的Critic和Actor网络，在交互过程中通过学习策略达到回报最大化，是探索和利用的过程。
    - **进化策略：** 两代(1+1)-ES、每代(1+1)-ES、(1+1)-ES+CEM等多种形式的进化策略，直接优化线上转化率目标。
  - **融合评估：** 重排算法的AB实验和效果评估，通过离线指标和线上指标，从匹配性、活跃性、多样性等方面，评估推荐结果的准确性和个性化程度，提高用户满意度。


---

#### 项目二：排序算法

- **项目技术：** XGBoost + DeepFM + Focal Loss + PointWise + PairWise + MMoE + PLE + TFRA + AUC + MAP + TOPN
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **项目简介：** 排序层的主要任务是对召回层产生的候选物品进行精细排序。排序层关注深度，通过先进的机器学习和深度学习模型，对每个候选物品的相关性进行评分，并根据得分进行排序。项目采用XGBoost和DeepFM等模型，通过Focal Loss提高模型的鲁棒性。采用多任务学习框架（MMoE、PLE），结合PointWise和PairWise两种排序方式，优化多目标排序效果。
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **职责描述：**
  - **样本权重：** 采用XGBoost和DeepFM等模型，通过设置样本权重，来控制模型多个目标的倾向，样本权重根据样本比例以及模型离线评价指标来调整。使用Focal Loss调整样本权重，处理类别不平衡问题，增强难样本的权重，提高模型的鲁棒性。
  - **多任务学习：** 结合查看、开聊、回复、达成、拒绝等多种目标学习，具体方式有：
    - 排序PairWise模型，让模型学习各个label样本pair对之间的相对顺序。选择优化目标，调整样本pair对的组合方式，根据item1和item2的label，进行损失加权，起到多个目标调和的作用。
    - 应用多任务学习框架，ESMM将CTR和CVR结合，解决CVR数据稀疏性问题；MMoE通过多个专家网络和门控机制处理多任务学习问题；PLE通过渐进式分层提取特征实现多任务学习，在模型结构上引入专门的任务特定层和共享层，逐步提取和分离特征，优化排序模型的多目标效果。
  - **ID特征：** 构建和优化userID和itemID的动态嵌入特征，提升模型的表达能力。利用TFRA的ID Dynamic Embedding技术，捕捉用户和物品的个性化特征，提高排序模型的表现。
  - **模型评估：** 使用AUC、GAUC（个性化排序，消除请求偏差）、MAP（位置顺序敏感）、TOPN（位置顺序敏感）等指标，评估排序模型的效果，拟合线上的实际效果，便于模型调整优化。


---

#### 项目三：召回算法

- **项目技术：** ALS SimItem + Content SimItem + Item2Vec Attr + Swing + Mind + U2U2I + U2I2I + UserCF + ItemCF + 策略召回
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **项目简介：** 召回层的主要任务是从庞大的物品库中筛选出一个相对较小的候选物品集合。这层主要关注广度，通过高效的检索算法迅速找到可能与用户相关的物品。项目结合多种召回技术，通过ALS、Content、Item2Vec等方式计算物品向量，实现相似物品召回。采用U2U2I、U2I2I等方法，计算user-item相似度。各路子召回，通过顺序融合、蛇形融合、自动融合等方式，提高召回效果。
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **职责描述：**
  - **SimItem：** 利用物品相似度进行物品召回，根据物品的共现关系找到相似物品，具体方式有：
    - **ALS：** 选择用户的查看和开聊行为，根据userid、itemid、score进行ALS计算。通过ALS分解用户-物品交互矩阵，生成用户和物品的隐向量表示，用于计算相似物品。
    - **Content：** 根据物品属性信息匹配度合计总分，包括距离，相似度，薪资、年龄、学历、公司匹配度等。通过属性信息来计算，扩大相似物品资源池，提高召回覆盖度。
    - **Item2Vec：** 基于用户行为得到itemid序列，增加side information，如city, position, title等，构造两两pair对item，利用n-gram特征捕捉词内部字符信息，训练fasttext模型，得到item向量。
    - **Swing：** user-item-user的结构比itemCF的单边结构更稳定，itemCF考察两个物品重合的受众比例有多高，如果很多用户同时喜欢两个物品，判定两个物品相似。Swing额外考虑重合的用户是否来自同一个小圈子，如果overlap(u1, u2)越大，就越要降低权重。
  - **向量召回：** Mind根据用户行为和用户画像特征，通过胶囊网络生成多个用户兴趣向量。U2U2I从用户到用户再到物品，U2I2I从用户到物品再到物品，结合用户行为序列，计算向量相似度，实现向量召回。
  - **策略召回：** 基于距离、相似度、活跃度等指标的策略召回，新物品召回，热门物品召回，冷启动物品召回，根据历史行为和偏好的个性化召回等策略。


---

#### 项目四：样本特征

- **项目技术：** Hive + Hadoop + Spark + Flink + Pearson Correlation + SHAP + Label + 特征分析 + 特征挖掘 + 特征选择
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **项目简介：** 数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。为了提升推荐系统的效果，本项目旨在构建高质量的样本特征表，准备模型训练数据。结合实际业务，开发特征工程，根据特征分析结果，针对性地选择特征，训练优化模型。项目的目标不仅是提高模型的预测准确性，还通过完善BI体系指标，为业务决策提供强有力的数据支持。
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **职责描述：**
  - **数据仓库：** 根据曝光日志、特征日志、行为日志等数据源，使用Hadoop和Spark大数据框架，构建全面的数据处理流程，包括清洗数据和ETL工程。通过提取特征和样本，建立高效的数据仓库。确保数据的高质量和一致性，为后续的特征工程和模型训练提供坚实的数据基础。
  - **样本指标：** 结合实际业务，规范标签体系。增加负向行为指标（如拒绝），细化正向行为指标（如已读），以提高指标的全面性和准确性。将新的标签体系更新到BI系统，并同步更新训练样本中的标签，以确保模型的准确性和稳定性。
  - **特征工程：** 深入挖掘和开发超过1500个特征，包括BC两端的属性特征、交叉特征（如匹配特征、偏好特征）、内容特征、统计特征、行为特征、序列特征、时间特征、以及距离、相似度和活跃度特征等。涵盖user-item的各个方面，为模型提供丰富的信息。
  - **特征选择：** 根据特征分析选择特征，具体方式有：XGB特征重要性、皮尔逊相关系数、特征排列重要性（去掉某个特征）、LR特征重要性（添加某个特征）、SHAP等。通过特征选择，去除冗余特征，提高模型的泛化能力，减少过拟合风险。



[//]: # (- 特征选择：)
[//]: # (  - XGB特征重要性：特征重要性值越大，特征越重要)
[//]: # (  - 皮尔逊相关系数：正值正相关，负值负相关；绝对值越大，相关性越大，跟模型无关)
[//]: # (  - 特征排列重要性：去掉某个特征，评估模型auc衰减得越多，特征越重要，跟模型有关)
[//]: # (  - LR特征重要性：添加某个特征，评估模型auc，auc涨得越多，特征越重要，跟模型有关)



#### 项目五：蓝领交付

- **项目技术：** Hive + Hadoop + Spark + BI + Feature + Label + DeepFM + FM + Gurobi
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **项目简介：** 针对自营模式的项目职位，通过曝光、喂奶、公海再激活等方式，候选人和职位开聊达成后生成线索，将线索分配给交付专员。交付专员负责跟进整个招聘过程，包括拨打电话、接通、约面、到面、面通、入职以及后续的离职管理。项目旨在通过推荐系统和模型优化，提升蓝领交付的效率和质量，提高线索转化率和用户满意度。
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **职责描述：** 
  - **样本特征：** 结合流量侧和分配侧的数据流，构建样本特征表，可视化BI漏斗指标。开发匹配特征、统计特征、偏好特征等。通过特征分析，挖掘用户行为和偏好，提高模型的预测准确性。
  - **推荐模型：** 构建一个推荐系统来优化交付流程，确保线索资源的最佳分配，包括分级层、分配层、全局匹配层、标签层等。具体内容有：
    - **分级层：** 线索分级模型，优先分配好线索，提高线索的转化率。以约面、到面、入职为目标，模型从DeepFM迭代到FM，调整超惨，提高模型的准确性。
    - **分配层：** 通过提高专员和线索的匹配度，提高线索的转化率。根据专员的历史表现和线索的属性特征，构建分配模型，提高线索的匹配度。
    - **全局匹配层：** 在全局范围内优化匹配，确保总效益最大化，用整数规划Gurobi方法实现。最大化匹配评分，约束条件有可推荐关系约束、专员库容约束、线索分配次数约束。
    - **标签层：** 通过给优质线索打上显示标签，加强专员的重视程度，提高线索的转化率。展示线索的活跃度、匹配度、优先级等标签，提高专员的工作效率。
  - **策略优化：** 针对实时线索和待分配线索，采用不同的分配策略。在分配时间、分配次数、线索流转等方面，结合具体业务，优化分配策略，提高线索的转化率和用户满意度。


---

<div style="display: flex; justify-content: space-between;">
    <p style="font-size: 20px;"><strong>算法工程师 — 必要科技有限公司</strong></p> 
    <p style="font-size: 20px;"><strong>2016 — 2018</strong></p>
</div>

担任电商平台算法工程师，负责用户画像、数据分析及推荐系统的设计和优化工作。通过构建精准的用户画像和深入的数据分析，优化推荐算法和个性化策略，提高推荐系统的准确度和效率。根据用户标签进行精准营销，提升购买率，减少流失率。通过持续的模型优化和策略调整，提升业务目标和用户体验。



#### 项目一：推荐系统

---


#### 项目二：用户画像






用户画像 
    - 收集和整合用户在平台上的行为数据（如浏览、点击、购买历史），建立详细的用户画像。 
    - 使用聚类分析、主成分分析等方法，对用户进行细分，提取用户偏好和特征。 
    - 基于用户特征进行用户分群，识别高价值用户和潜在流失用户，制定针对性的运营策略。

数据分析 
    - 分析用户行为数据，挖掘影响用户购买决策的关键因素，帮助产品团队优化用户体验。 
    - 使用数据可视化工具（如Tableau），将分析结果以图表形式呈现，供管理层决策参考。 
    - 定期生成数据报告，提供用户行为趋势和市场反馈。

推荐系统 
    - 设计并实现协同过滤、随机森林、XGBoost等多种推荐算法，提升推荐系统的准确性和多样性。 
    - 引入实时数据流处理框架（如Kafka、Flink），实时推荐，提高推荐的时效性。 
    - 通过AB测试评估不同推荐策略的效果，持续优化推荐模型。

  - 用户画像系统的完善使得用户运营更加精准，用户粘性和满意度显著提升。 
  - 通过深入的数据分析，优化了平台的商品展示和库存管理，提高了销售转化率和库存周转率。 
  - 推荐系统的优化显著提升了推荐的点击率和转化率，带动了平台的整体销售增长。

- 这些工作，不仅提高了电商平台的推荐效果和用户体验，还帮助公司更好地理解用户需求，制定更精准的市场策略。
