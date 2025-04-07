---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Video, How Do Your Tokens Merge?
---

# Video, How Do Your Tokens Merge?

[Sam Pollard](https:/sjpollard.github.io), [Michael Wray](https:/mwray.github.io)

University of Bristol

eLVM@CVPR'25


![Intro](assets/intro.png)

<br>

![Method](assets/method.png)

# Abstract

Video transformer models require huge amounts of compute resources due to the spatio-temporal scaling of the input. Tackling this, recent methods have proposed to drop or merge tokens for image models, whether randomly or via learned methods. Merging tokens has many benefits: it can be plugged into any vision transformer, does not require model re-training, and it propagates information that would otherwise be dropped through the model. Before now, video token merging has not been evaluated on temporally complex datasets for video understanding. In this work, we explore training-free token merging for video to provide comprehensive experiments and find best practices across four video transformers on three datasets that exhibit coarse and fine-grained action recognition. Our results showcase the benefits of video token merging with a speedup of around 2.5X while maintaining accuracy (avg. -0.55% for ViViT).

# Links
[Code](https://github.com/sjpollard/video-how-do-your-tokens-merge) | [Arxiv]()

# Bibtex

```

```
