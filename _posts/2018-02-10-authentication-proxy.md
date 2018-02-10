---
layout: post
title: "Death by Proxy"
date: 2018-02-10 12:25:00 +0000
categories: open-source software
img: 20180210/title.jpg
---

# Death by Proxy

I recently started a new job, in a completely new country. I've so far been at the company for just about a week.

It has been interesting, and I think will continue to be interesting, to observe and feel the impact of different data science maturity levels, as well as the general organisational mindset when it comes to data. I will definitely be writing about this in future posts.

 Having come from an online software platform company, where technology access was laissez-faire and a massive amount of freedoms were enjoyed, it has been frustrating to say the least to deal with a legacy corporation's policies: locked down PCs with no administrator access, no easy access to *nix based compute resources (I have always felt sorry for people who had no choice but to use Cygwin...now I'm one of those people TT), but by far the worst thing is an overly restrictive corporate proxy.
 
 I understand the reasoning of corporate proxies in general, but not when they break pretty much every tool I use as a data scientist on a daily basis. Let's go through the list:

 - jupyter notebooks 
 - pip
 - condas
 - git
 - curl, wget, etc
 
 I literally had to note to myself all the things I needed and wait to install them when I got home on most days. Talk about productivity killing.
 
 Well now, I hear you ask, how can a corporate proxy be **so** bad?
 
 Well, for this particular proxy there are two main reasons:

 1. A badly configured PAC (Proxy Auto Configuration) file. 
 2. Only allowing a Kerberos authentication.
 
 What's a PAC? I hear you ask. Well, when you set up your internet proxy settings, there's generally an option to "auto-configure" by linking to a PAC file. This file is just some javascript and contains essentially a bunch of "if-else" rules for assigning proxy endpoints based on the request being sent.
 
 Usually, internal company requests will be routed directly, and only external requests get assigned to a proxy endpoint.
 
 However, for whatever reason in this PAC file corporate IT had not properly specified settings for localhost (or 127.0.0.1), which leads to jupyter notebooks breaking, as when you open it in your browser, the localhost requests that the browser makes get proxied and therefore cannot connect successfully to the kernel.
 
 This was easily rectified, which gave me jupyter notebooks back, but the second issue was a bit a trickier to tackle.
 
In a corporate environment, proxies can generally be authenticated to in 3 ways: Basic, NTLM and Kerberos, usually you would send a proxy connect request that allows failover to any one of the 3 methods if it doesn't work (the Proxy-Negotiate method), however, for security reasons all but Kerberos authentication had been disabled on this corporate proxy.

This was a big issue because most of these command lines tools are only able to deal with Basic authentication schemes. Even allowing NTLM would have been fine too as a quick google search for "NTLM Authentication Proxy" will yield you a few things you can run (namely CNTLM).

Unfortunately, Kerberos is a bit of a trickier beast to tackle, and through ~~hours~~ days of Googling (or Bing-ing, as I have to do in China), I could not find any full working solutions to deal with a Kerberos authenticated proxy on Windows.

The main problem to dealing with Kerberos authentication is that on a Windows environment, Microsoft has built their own implementation of the Kerberos authentication scheme, exposing an API called MS-SSPI for fetching Kerberos tickets. This is compared to the otherwise standard GSSAPI which the standard MIT implementation uses.

After really searching hard to find a full solution (most I found didn't work at all, and the working one did not support HTTPS requests (and was written in C#, eugh)) and failing, I had to end up rolling my own, which I did (and learned some Go in the process).

I hope this will help some other poor souls out there who are in the same boat as me. [https://github.com/qichaozhao/authentication-proxy].

Also, if you want to talk to me about TCP, TLS Handshakes and Certificates, Kerberos, the net/http module in Go, anything about proxies, or how amazing libcurl is, I'm good for it now. :P

-qz