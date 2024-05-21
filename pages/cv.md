---
layout: page
title: CV
description: 个人简介
keywords: CV
comments: true
menu: CV
permalink: /cv/
---


<ul class="flex-container">
{% for website in site.data.social %}
<li>{{website.sitename }}：<a href="{{ website.url }}" target="_blank">{{ website.name }}</a></li>
{% endfor %}
{% if site.url contains 'huzixia.github.io' %}
<li>公众号：AI Freedom <br /></li>
{% endif %}
</ul>


## 基本信息

- **姓名：胡紫霞**
- **教育：中国农业大学 985硕士**
- **MetaGPT 代码贡献者**

---

- **人工智能-大模型与AIGC 高级证书**
- **大数据分析师 高级证书**
- **AI Agent Developer证书**

---

- **动手学 AI 视频生成 最佳视频奖**
- **Multi-Agent for X AI创客松 最佳人气奖**




## 专业技能


{% for skill in site.data.skills %}
### {{ skill.name }}
<div class="btn-inline">
{% for keyword in skill.keywords %}
<button class="btn btn-outline" type="button">{{ keyword }}</button>
{% endfor %}
</div>
{% endfor %}

### 算法工程师

### 数据分析师

### 多模态算法工程师

### 大模型开发工程师


## 工作经验

<div style="display: flex; justify-content: space-between;">
    <h3>算法工程师 — 北京华品博睿网络技术有限公司</h3> <span style="text-align: right">2018 — 2023</span>
</div>
- 店长直聘，数据分析，推荐算法
- BOSS直聘，数据分析，推荐算法
- 蓝交付，数据分析，推荐算法



<div style="display: flex; justify-content: space-between;">
    <h3>算法工程师 — 北京海纳金川科技有限公司</h3> <span style="text-align: right">2023 — 2024</span>
</div>
- 大模型开发，多模态大模型算法


## 项目经历

### 推荐算法，召回、粗排、精排、重排

- **召回：** 召回
- **粗排：** 粗排
- **精排：** 精排
- **重排：** 重排

### 数据分析，用户画像，用户行为，生命周期，商业增长

- **用户画像：** 用户画像
- **用户行为：** 用户行为
- **生命周期：** 生命周期
- **商业增长：** 商业增长


### 多模态大模型


### 大模型开发



