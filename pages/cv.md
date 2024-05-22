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


## **基本信息**


{% for base in site.data.base %}
### {{ base.name }}
<div class="btn-inline">
{% for base in base.keywords %}
<button class="btn btn-outline" type="button">{{ base }}</button>
{% endfor %}
</div>
{% endfor %}



{% if site.skill_software_keywords %}
<div class="panel panel-default">
    <div class="panel-heading">
        <h3 class="panel-title">Software Engineer Keywords</h3>
    </div>
    <div class="panel-body">
        {% for keyword in site.skill_software_keywords %}
        <button class="btn btn-default" type="button">{{ keyword }}</button>
        {% endfor %}
    </div>
</div>
{% endif %}


{% for base in site.data.base %}
<div class="panel panel-default">
    <div class="panel-heading">
        <h3 class="panel-title">身份</h3>
    </div>
    <div class="panel-body">
        {% for base in base.keywords %}
        <button class="btn btn-default" type="button">{{ base }}</button>
        {% endfor %}
    </div>
</div>
{% endif %}




## **专业技能**


{% for cv in site.data.cv %}
### {{ cv.name }}
<div class="btn-inline">
{% for cv in cv.keywords %}
<button class="btn btn-outline" type="button">{{ cv }}</button>
{% endfor %}
</div>
{% endfor %}



## **工作经验**


<div style="display: flex; justify-content: space-between;">
    <p><strong>算法工程师 — 北京海纳金川科技有限公司</strong></p> 
    <p><strong>2023 — 2024</strong></p>
</div>

- 大模型开发，多模态大模型算法



<div style="display: flex; justify-content: space-between;">
    <p><strong>算法工程师 — 北京华品博睿网络技术有限公司</strong></p> 
    <p><strong>2018 — 2023</strong></p>
</div>

- 店长直聘，数据分析，推荐算法
- BOSS直聘，数据分析，推荐算法
- 蓝交付，数据分析，推荐算法



## **项目经历**


### 多模态大模型


### 大模型开发


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







