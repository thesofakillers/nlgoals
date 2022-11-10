---
documentclass: article
title: Natural Language Interfaces and Reward Hacking in RL
subtitle: "Research Proposal"
date: \today
author: Giulio Starace
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

<!-- The following is the research proposal for my AI master's thesis at the -->
<!-- University of Amsterdam. The topic has been accepted. My internal supervisor is -->
<!-- Niklas Höpner of the AMLab, who has seconded my application to CHAI. Part-time -->
<!-- work in the form of a literature review and brainstorming has begun in early -->
<!-- November, with full-time work commencing in February and an expected completion -->
<!-- date of August $25^{th}$, 2023. There will be a month-long pause in the work in -->
<!-- December due to a course I will be attending full-time. -->

<!-- At CHAI, I hope to receive mentorship from experts in AI Safety (AIS). While my -->
<!-- local supervision is of great value, the staff at my institute are not very -->
<!-- familiar with AIS. I hope that my CHAI mentor would be able to guide me in -->
<!-- making the right mental connections and finding appropriate citations that I may -->
<!-- have otherwise missed. I also hope to receive more classical mentorship in the -->
<!-- form of additional perspectives and creative approaches to the problem. -->
<!-- Ultimately the goal is to produce a piece of research worthy of peer review and -->
<!-- publication and to connect with more people in AIS. AIS is the direction I would -->
<!-- like to pursue in my career. -->

<!-- Finally, I should note that I am generally curious about alternative approaches -->
<!-- to Reinforcement Learning that address the issue of reward hacking. I am also -->
<!-- interested in human-AI interface design, currently seemingly dominated by -->
<!-- prompting which is where most of my experience lies outside of AI safety. I -->
<!-- developed the proposal below because it captured both interests while remaining -->
<!-- flexible in terms of what can be contributed and to potential topic pivots. I -->
<!-- should note that both my supervisor and I are open to adapting the topic to -->
<!-- something similar that may be more suitable for CHAI mentorship. -->

<!-- \newpage -->

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
[@evans_learning_2015;@evans_learning_2016]. IRL is often also criticised for
overlooking side-effects [@krakovna_penalizing_2019] and encouraging
power-seeking [@turner_optimal_2022]. Regardless of these issues, IRL does not
necessarily address the overarching problem, as reward hacking has also been
observed in this context [@ibarz_reward_2018].

In general, IRL is considered to be a subfield of _**imitation learning**_
[@abbeel_apprenticeship_2004], where the goal is now to predict trajectories,
given expert demonstrations. Imitation learning faces similar limitations to
those of IRL.

### Preference-based learning

Preference-based learning circumvents the need for demonstrations by using a
more direct signal of human preferences. This includes, for example, directly
asking users what they want via e.g. pairwise comparisons
[@christiano_deep_2017; @sadigh_active_2017; @biyik_batch_2018;
@lee_pebble_2021]. This however can be limited in expressivity: consider a case
in which two sub-optimal but complimentary correct trajectories are presented.
Under pairwise comparison, there is no way to express the necessary granularity
in preferences for this example. Another potential issue is that the _how_ is
difficult to express.

## Proposed approach

Using advances in natural language processing, particularly in large language
models (LLMs) [@vaswani_attention_2017; @brown_language_2020;
@sanh_multitask_2022] and prompting techniques [@gal_image_2022;
@reynolds_prompt_2021; @wu_ai_2022], and inspired by their applications beyond a
pure NLP context [@dosovitskiy_image_2022; @rombach_high-resolution_2022;
@ramesh_hierarchical_2022], we can develop a more natural interface between
human and machine to specify goals and or rewards. This is after-all how humans
communicate desired outcomes to each other. There already exist many recent
works leveraging the expressivity of language models in an RL context
[@zhou_inverse_2021; @sumers_learning_2021; @jiang_vima_2022;
@shridhar_cliport_2021; @huang_inner_2022; @shridhar_perceiver-actor_2022;
@ding_robot_2022; @yao_react_2022; @brooks_-context_2022;
@gramopadhye_generating_2022; @yu_using_2022; @lu_neuro-symbolic_2022;
@watkins_teachable_2021]. A number of NL-RL-hybrid environments and datasets
[@anderson_vision-and-language_2018; @liu_reinforcement_2022;
@jiang_yunfan_vima_2022; @zholus_iglu_2022; @fan_minedojo_2022] have accompanied
many of these papers in the field. These works however mostly focus on their
contributions to planning performance, learning efficiency and other more common
RL metrics of success. Using techniques similar to those developed by
@pan_effects_2022 and @lee_pebble_2021 and taking inspiration from the recent
works cited above, this work hopes to explore the question: **to what extent can
natural language interfaces curtail the issue of reward hacking in RL?**

## Potential outcomes and focus points

This proposal has mostly motivated the problem from the perspective of
Reinforcement Learning. In this sense, one of the potential outcomes to focus on
is an alternative method for addressing reward misspecification and hacking in
RL.

However, the research could also pivot towards investigating ways of improving
the ability of language models in understanding human instructions. This would
be similar to the work of @scheurer_training_2022, with perhaps more attention
to the safety implications. Prompts currently dominate interfaces through which
humans can guide capabilities, but can we develop better and safer interfaces?
Human-AI interface design may be a fruitful area of AI alignment research.

A sub-problem that needs addressing in integrating language feedback into an RL
policy is that of grounding the language to the environment. A contribution in
this area could also be made, although it is not immediately clear how this
directly contributes to safety.

Of course, the holy grail would be contributions to all of these areas in a
unified solution. By casting a wider net at the beginning, the hope is that at
least one of these can be achieved.
