---
layout: page
title: About
description: 万年太久，只争朝夕，活在当下。
keywords: huzixia
comments: true
menu: 关于
permalink: /about/
---

我是胡紫霞，RedHerring。

万年太久，只争朝夕，活在当下。

Follow your heart, keep learning.


## 联系

<ul>
{% for website in site.data.social %}
<li>{{website.sitename }}：<a href="{{ website.url }}" target="_blank">@{{ website.name }}</a></li>
{% endfor %}
{% if site.url contains 'huzixia.github.io' %}
<li>
微信公众号：AI Freedom <br />
<img style="height:192px;width:192px;border:1px solid lightgrey;" src="{{ site.url }}/assets/images/qrcode.jpg" alt="AI Freedom" />
</li>
{% endif %}
</ul>


## Skill Keywords

{% for skill in site.data.skills %}
### {{ skill.name }}
<div class="btn-inline">
{% for keyword in skill.keywords %}
<button class="btn btn-outline" type="button">{{ keyword }}</button>
{% endfor %}
</div>
{% endfor %}




[//]: # ()
[//]: # ()
[//]: # (## 捐助)

[//]: # ()
[//]: # (做一些微小的事情，如果对你有帮助，可以考虑请我喝杯咖啡。)

[//]: # ()
[//]: # (Did some tiny things, consider buying me a cup of coffee if it helps you.)

[//]: # ()
[//]: # (## Wechat)

[//]: # ()
[//]: # (<img style="width:256px;border:1px solid lightgrey;" src="{{ site.url }}/assets/images/receipt-code-wechat.jpg" alt="wechat receipt code" />)

[//]: # ()
[//]: # ()
[//]: # (## Alipay)

[//]: # ()
[//]: # (<img style="width:256px;border:1px solid lightgrey;" src="{{ site.url }}/assets/images/receipt-code-alipay.jpg" alt="alipay receipt code" />)
