---
title: Natural Language Interfaces for Specification Learning
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
  - top=1in
  - left=1.5in
  - right=1.5in
  - bottom=1.5in
header-includes:
  - \pagenumbering{gobble}
  - \usepackage{setspace}
---

## The Problem

In Reinforcement Learning (RL), the goal of a particular autonomous agent is
formalised in the form of a reward signal emitted by the environment to the
agent. This reward signal is typically computed via some handcrafted reward
function. However, handcrafted reward functions can be difficult to specify for
more complex problems and environments, and can lead to undesired agent
behaviour due to reward hacking [@pan_effects_2022].

## Why We Want To Solve It

Relying on handcrafted reward functions can be tedious, requiring at times ample
domain knowledge and mental effort. Furthermore, after design, the reward
function has to be manually implemented as part of the agent’s environment.
Finally, handcrafted reward functions may suffer from bias and human error,
leading to subpar or undesired performance of our models. Generally, these are
symptoms signaling difficulty in scaling and generalisation. In the case of
undesired model performance, this has safety implications
[@hendrycks_unsolved_2022].

## Current Solutions and their Shortcomings

### Inverse Reinforcement Learning

Inverse Reinforcement Learning (IRL) [@ng_algorithms_2000;
@ziebart_maximum_2008; @ho_generative_2016] is the problem of extracting a
reward function given observed expert behaviour (demonstrations). While
promising and perhaps suitable for many problems, IRL presents some limitations:

- Expert demonstrations are not always available and can be difficult to obtain
- For many environments it is very difficult to determine the reward function
  from the demonstrations.
  - There is some research addressing this issue
    [@choi_inverse_2011;@amin_towards_2016].
- Model performance may be limited to the performance of the experts from which
  it is learning.
  - There is some research addressing this issue
    [@evans_learning_2015;@evans_learning_2015-1].
- Natural intelligent agents (e.g. humans) don’t always need expert
demonstrations to learn a reward function, so this is indicative of a lack of
generalisation.
<!-- - [Model Mis-specification and Inverse Reinforcement Learning | Academically Interesting (wordpress.com)](https://jsteinhardt.wordpress.com/2017/02/07/model-mis-specification-and-inverse-reinforcement-learning/) -->

IRL has a considerable overlap with _**imitation learning**_
[@abbeel_apprenticeship_2004], where the goal is now to predict trajectories,
given expert demonstrations. Imitation learning faces similar limitations to
those of IRL.

### Preference-Based Learning

Preference-based learning circumvents the need for demonstrations by using a
more direct signal of human preferences. This includes, for example, directly
asking users what they want via e.g. pairwise comparisons
[@christiano_deep_2017; @sadigh_active_2017; @biyik_batch_2018].

- Expression of preferences via pairwise comparison can be limited.
- Expressed preferences may be different from real preferences.

## Proposed Approach

Using advances in natural language processing, particularly in large language
models (LLMs) [@vaswani_attention_2017; @brown_language_2020;
@sanh_multitask_2022] and prompting techniques [@gal_image_2022;
@reynolds_prompt_2021; @wu_ai_2022], and inspired by their applications beyond a
pure NLP context [@dosovitskiy_image_2021; @rombach_high-resolution_2022;
@ramesh_hierarchical_2022], we can develop a more natural interface between
human and machine to specify goals and or rewards. This is after-all how humans
communicate desired outcomes to each other.
