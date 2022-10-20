---
documentclass: article
title: Natural Language Interfaces and Reward Hacking in RL
subtitle: UvA MSc AI - Thesis Proposal
date: \today
author: Giulio Starace - 13010840
bibliography:
  - ../references.bib
csl:
  - acm.csl
reference-section-title: References
link-citations: true
colorlinks: true
geometry:
  - top=1.5in
  - left=1.5in
  - right=1.5in
  - bottom=1.5in
header-includes:
  - \pagenumbering{gobble}
  - \usepackage{setspace}
---

## The problem

In Reinforcement Learning (RL), the goal of a particular autonomous agent is
formalised in the form of a reward signal emitted by the environment to the
agent. This reward signal is typically computed via some handcrafted reward
function. However, handcrafted reward functions can be difficult to specify for
more complex problems and environments, and can lead to undesired agent
behaviour due to reward hacking [@pan_effects_2022;@skalse_defining_2022].

## Why we want to solve it

Addressing the issue of reward misspecification is important because it is one
of the many limiting factors that make RL difficult to apply. Furthermore, due
to reward hacking, the issue can lead to undesired behaviour. The negative
impacts of misbehaviour can be as simple as a model underperforming in
production and as dire as causing safety concerns [@hendrycks_unsolved_2022].

## Current solutions and their shortcomings

### Inverse Reinforcement Learning

Inverse Reinforcement Learning (IRL) [@ng_algorithms_2000;
@ziebart_maximum_2008; @ho_generative_2016] is the problem of extracting a
reward function given observed expert behaviour (demonstrations). While
promising and perhaps suitable for many problems, IRL has some limitations. For
instance expert demonstrations are not always available and can be difficult to
obtain. Furthermore, for many environments it is very difficult to determine the
reward function from the demonstrations [@choi_inverse_2011;@amin_towards_2016].
Another limitation is that model performance may be limited to the performance
of the experts from which it is learning
[@evans_learning_2015;@evans_learning_2015-1]. IRL is often also criticised for
overlooking side-effects [@krakovna_penalizing_2019] and encouraging
power-seeking [@turner_optimal_2021]. Even if these issues were addressed, IRL
does not necessarily address the overarching problem, as reward hacking has been
observed in the IRL context as well [@ibarz_reward_2018].

In general, IRL is considered to be a subfield of _**imitation learning**_
[@abbeel_apprenticeship_2004], where the goal is now to predict trajectories,
given expert demonstrations. Imitation learning faces similar limitations to
those of IRL.

### Preference-based learning

Preference-based learning circumvents the need for demonstrations by using a
more direct signal of human preferences. This includes, for example, directly
asking users what they want via e.g. pairwise comparisons
[@christiano_deep_2017; @sadigh_active_2017; @biyik_batch_2018]. The main
approach is to express preferences via pairwise comparison. This however can be
limited in expressivity: consider a case in which two sub-optimal but
complimentary correct trajectories are presented. Under pairwise comparison,
there is no way to express the necessary granularity in preferences for this
example. Another potential issue is that the expressed preferences may be
different from the real preferences.

## Proposed approach

Using advances in natural language processing, particularly in large language
models (LLMs) [@vaswani_attention_2017; @brown_language_2020;
@sanh_multitask_2022] and prompting techniques [@gal_image_2022;
@reynolds_prompt_2021; @wu_ai_2022], and inspired by their applications beyond a
pure NLP context [@dosovitskiy_image_2021; @rombach_high-resolution_2022;
@ramesh_hierarchical_2022], we can develop a more natural interface between
human and machine to specify goals and or rewards. This is after-all how humans
communicate desired outcomes to each other. There already exist many recent
works leveraging the expressivity of language models in an RL context
[@zhou_inverse_2020; @sumers_learning_2021; @jiang_vima_2022;
@shridhar_cliport_2021; @huang_inner_2022; @shridhar_perceiver-actor_2022;
@ding_robot_2022; @yao_react_2022; @brooks_-context_2022;
@gramopadhye_generating_2022; @yu_using_2022; @lu_neuro-symbolic_2022]. A number
of NL-RL-hybrid environments and datasets [@anderson_vision-and-language_2018;
@liu_reinforcement_2018; @jiang_yunfan_vima_2022; @zholus_iglu_2022;
@fan_minedojo_2022] have accompanied many of these papers in the field. These
works however mostly focus on their contributions to planning performance,
learning efficiency and other more common RL metrics of success. Using
techniques similar to those developed by @pan_effects_2022 and taking
inspiration from the recent works cited above, this work hopes to explore the
question: **to what extent can natural language interfaces curtail the issue of
reward hacking in RL?**
