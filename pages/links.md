---
layout: page
title: Links
description: 没有链接的博客是孤独的
keywords: 工具链接
comments: true
menu: 链接
permalink: /links/
---

> God made relatives. Thank God we can choose our friends.

<ul>
{% for link in site.data.links %}
  {% if link.src == 'life' %}
  <li><a href="{{ link.url }}" target="_blank">{{ link.name}}</a></li>
  {% endif %}
{% endfor %}
</ul>

> 工具链接

<ul>
{% for link in site.data.links %}
  {% if link.src == 'www' %}
  <li><a href="{{ link.url }}" target="_blank">{{ link.name}}</a></li>
  {% endif %}
{% endfor %}
</ul>


Hi，您是第 ![Visitor Count](https://profile-counter.glitch.me/HuZixia/count.svg) 位访问者！
