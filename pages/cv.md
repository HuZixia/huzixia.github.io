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
  - **模型微调：** 开发优化模型代码，适配LoRA、P-Tuning V2和QLora等微调技术，调整参数，对ChatGLM和LLaMA模型进行微调训练。集成DeepSpeed技术，实现模型的分布式训练，加快训练速度，提高资源利用率。
  - **模型评估：** 评估模型微调的效果，针对 Function Calling 的每个参数（即 Slot），评估准确率、召回率、F1 值；针对文本回复，评估输出文本与参考文本之间的 BLEU Score。

---

#### 项目二：智能客服

- **项目技术：** RAG + LangChain + Function Calling + Text Embedding + Chromadb + Hybrid Search + ChatGLM + GPT-4V
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **项目简介：** 为了提高用户体验和服务效率，本项目设计开发了一个基于大语言模型的智能客服系统。希望能够准确理解用户问题，并提供相关的解决方案和建议。通过构建一个包含常见问题答案、操作指南和服务流程的知识库，利用RAG技术，从知识库中检索相关信息，生成高质量的响应，结合Hybrid Search方法提高检索效率和准确性。此外，系统还具有实时监控和分析功能，确保稳定性。
<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **职责描述：**
  - **系统设计：** 设计智能客服系统的整体架构，确保系统能够准确响应用户问题。包括选择和配置合适的大模型、设计系统模块和数据流，以及集成各种功能组件。
  - **知识库构建：** 收集、整理和存储知识信息，包括常见问题答案、操作指南和服务流程等。确保知识库信息准确性和及时更新。
  - **RAG集成：** 实现RAG技术，根据用户输入，通过向量检索模型和排序模型，从知识库获取参考答案。将用户输入和参考答案一起提供给大语言模型，生成最终回复。
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
- **项目简介：** 本项目旨在开发一个智能视频生成系统，通过整合多种先进技术，实现从文本、图片和视频生成高质量视频内容。系统基于多模态大模型，结合Agent协作机制，支持从文本生成视频、从文字和图片生成视频，以及从文字和现有视频生成新的视频。项目利用SD、DALL-E、通义万象生成图像，使用TTS技术生成语音，采用Moviepy合成视频。根据用户反馈进行迭代优化，提升效果。

<style>
p {
    margin-bottom: 2em; /* 或者使用其他值，根据需要调整 */
}
</style>
- **职责描述：**
  - **系统开发：** 设计和开发视频生成系统，涵盖文字模块、音频模块、音乐模块、图像模块、视频模块等。集成各个模块，确保功能齐全，能够高效、准确地生成符合用户需求的视频内容。
  - **Agent协作：** 实现多个Agent的协作，确保不同模块的无缝衔接和高效运行，实现workflow多模态大模型协作，减少延迟和错误。
  - **功能模块：** 支持三种功能，文本生成视频，根据输入的文本生成剧本，进行分镜，然后生成图片，结合语音，合成视频；从文字和图片生成视频；从文字和现有视频生成新的视频。
  - **前后一致：** 在Prompt上进行微调，添加记忆功能，增强前后图片的角色一致性和内容连贯性，优化视频效果，提高用户满意度。


---

<div style="display: flex; justify-content: space-between;">
    <p style="font-size: 20px;"><strong>算法工程师 — 北京华品博睿网络技术有限公司</strong></p> 
    <p style="font-size: 20px;"><strong>2018 — 2023</strong></p>
</div>

在招聘平台工作期间，负责三大业务线：店长直聘、BOSS直聘、蓝交付。这些业务线的核心目标是通过智能推荐系统提升用户的招聘和求职体验。为了实现这一目标，我主导了标签体系、样本数据、特征工程、数据分析、推荐系统、监控体系等多个方面的工作。

主要任务是设计和优化三大业务线的推荐系统，包括推荐算法、召回、粗排、精排、重排等各个环节。此外，还负责数据分析、用户画像、用户行为分析、生命周期管理以及推动商业增长。



#### 项目一：重排算法

- **项目技术：** 
- **项目简介：** 
- **职责描述：**

---

#### 项目二：排序算法

- **项目技术：** 
- **项目简介：** 
- **职责描述：**

---

#### 项目三：召回算法

- **项目技术：** 
- **项目简介：** 
- **职责描述：**

---

#### 项目四：样本特征

- **项目技术：** 
- **项目简介：** 
- **职责描述：**




推荐算法 
    - 召回：设计和实现多种召回策略（Strategy、Item2Vec、Swing、Mind等），以提高推荐的覆盖率和多样性。 
    - 粗排：使用Xgboost和DeepFM等算法对召回结果进行粗排，保证推荐结果的质量。 
    - 精排：应用Xgboost、DeepFM、MMoE、PLE、TFRA、PointWise、PairWise、MTL等算法对粗排结果进行精细排序，进一步提升推荐的准确性。 
    - 重排：采用MTL、CEM、ES、RL、级联模型、融合模型等技术对最终的推荐结果进行重排，优化用户体验和业务目标。

样本体系 
    - 构建和管理大规模样本数据，确保数据的完整性和准确性。
    - 设计数据采集和处理流程，提升数据处理效率。

特征工程
    - 开发和优化多种特征，包括老板特征、牛人特征、职位特征、上下文特征、行为特征、偏好特征、兴趣特征等，提升模型的表现。 
    - 应用特征选择和特征生成技术，增强模型的泛化能力。

数据挖掘
    - 分析用户行为数据，构建用户画像，识别关键用户群体和行为模式。
    - 通过数据分析和挖掘，为产品优化和业务决策提供支持。

监控体系 
    - 设计和实施监控体系，实时监控推荐系统的性能和稳定性。 
    - 通过日志分析和异常检测，快速发现和解决问题。
    
商业增长 
    - 结合用户生命周期分析，制定个性化营销策略，推动用户活跃度和商业转化率的提升。
    - 与产品、运营团队紧密合作，推动产品迭代和业务增长。

  - 推荐系统的点击率和转化率显著提高，用户满意度大幅提升。
  - 数据分析和用户画像的应用，使产品优化和业务决策更加精准。
  - 商业增长显著，实现了用户活跃度和商业转化率的双重提升。

- 这些工作，不仅提升了推荐系统的效果，使得推荐结果更加符合用户需求，还增强了平台的核心竞争力，为公司带来了显著的商业收益。

---

<div style="display: flex; justify-content: space-between;">
    <p style="font-size: 20px;"><strong>算法工程师 — 必要科技有限公司</strong></p> 
    <p style="font-size: 20px;"><strong>2016 — 2018</strong></p>
</div>


担任电商平台算法工程师，负责用户画像、数据分析及推荐系统的设计和优化工作。

主要任务是通过用户画像和数据分析，提升推荐系统的精准度和效率，从而提高用户的购买率和满意度，降低用户流失率。


#### 项目一：推荐系统

---


#### 项目二：数据分析






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
