---
layout: page
title: CV
description: 个人简介
keywords: CV
comments: true
menu: CV
permalink: /cv/
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

## 基本信息

- **姓名：** 胡紫霞
- **教育：** 中国农业大学 985硕士
- **Code：** MetaGPT 代码贡献者

---

- **工信部：** 人工智能—大模型与AIGC 高级证书
- **工信部：** 大数据分析师 高级证书
- **Agent：** AI Agent Developer (MetaGPT & AgentScope)

---

- **AI创客松：** Multi-Agent for X 最佳人气奖 （阿里云 & 魔搭社区 & Datawhale）
- **开源社区：** 动手学 AI 视频生成 最佳视频奖 （Datawhale & 奇想星球）
- **企业荣誉：** 多次 荣获企业年度 A级 绩效奖
- **学校荣誉：** 多次 荣获国家奖学金、国家励志奖学金等



## **专业技能**


{% for cv in site.data.cv %}
#### {{ cv.name }}
<div class="btn-inline">
{% for info in cv.keywords %}
<button class="btn btn-outline" type="button">{{ info }}</button>
{% endfor %}
</div>
{% endfor %}



## **工作经历**


<div style="display: flex; justify-content: space-between;">
    <p style="font-size: 20px;"><strong>算法工程师 — 北京海纳金川科技有限公司</strong></p> 
    <p style="font-size: 20px;"><strong>2023 — 2024</strong></p>
</div>

- 大模型开发，多模态大模型算法



<div style="display: flex; justify-content: space-between;">
    <p style="font-size: 20px;"><strong>算法工程师 — 北京华品博睿网络技术有限公司</strong></p> 
    <p style="font-size: 20px;"><strong>2018 — 2023</strong></p>
</div>

- 店长直聘，数据分析，推荐算法
- BOSS直聘，数据分析，推荐算法
- 蓝交付，数据分析，推荐算法


<div style="display: flex; justify-content: space-between;">
    <p style="font-size: 20px;"><strong>算法工程师 — 必要科技有限公司</strong></p> 
    <p style="font-size: 20px;"><strong>2016 — 2018</strong></p>
</div>


- **S (Situation)**
    - 担任电商平台算法工程师，负责用户画像、数据分析以及推荐系统的设计和优化工作。

- **T (Task)**
    - 主要任务是通过用户画像和数据分析，提升推荐系统的精准度和效率，从而提高用户的购买率和满意度，降低用户流失率。

- **A (Action)**
  - **用户画像** 
    - 通过收集和整合用户在平台上的行为数据（如浏览、点击、购买历史），建立详细的用户画像。 
    - 使用聚类分析、主成分分析等方法，对用户进行细分，提取用户偏好和特征。 
    - 利用机器学习算法（如K-means聚类、决策树等），预测用户的行为和偏好。

  - **数据分析** 
    - 分析用户行为数据，挖掘影响用户购买决策的关键因素，帮助产品团队优化用户体验。 
    - 使用数据可视化工具（如Tableau），将分析结果以图表形式呈现，供管理层决策参考。 
    - 定期生成数据报告，提供用户行为趋势和市场反馈。

  - **推荐系统** 
    - 设计并实现协同过滤、随机森林、XGBoost等多种推荐算法，提升推荐系统的准确性和多样性。 
    - 引入实时数据流处理框架（如Kafka、Flink），实现实时推荐，提高推荐的时效性。 
    - 通过A/B测试评估不同推荐策略的效果，持续优化推荐模型。

- **R (Result)**
  - 用户画像的准确性显著提升，用户细分的精度提高了40%，推荐系统的个性化推荐效果明显改善。 
  - 通过精准的用户画像和数据分析，平台的推荐点击率提升了50%，用户平均停留时间增加了30%。 
  - 推荐系统优化后，平台的整体转化率提高了40%，用户流失率下降了20%。

- 通过这些工作，不仅提高了电商平台的推荐效果和用户体验，还帮助公司更好地理解用户需求，制定更精准的市场策略。



## **项目经历**


#### 多模态大模型

- **微调：** 微调
- **多模态：** 多模态
- **STAR描述**

#### 大模型开发

- **RAG：** RAG
- **Agent：** Agent


#### 推荐算法，召回、粗排、精排、重排

- **召回：** 召回
- **粗排：** 粗排
- **精排：** 精排
- **重排：** 重排

#### 数据分析，用户画像，用户行为，生命周期，商业增长

- **用户画像：** 用户画像
- **用户行为：** 用户行为
- **生命周期：** 生命周期
- **商业增长：** 商业增长







