



## The problem

In Reinforcement Learning (RL), the goal of a particular autonomous
agent is formalised in the form of a reward signal emitted by the
environment to the agent. This reward signal is typically computed via
some handcrafted reward function. However, handcrafted reward functions
can be difficult to specify for more complex problems and environments,
and can lead to undesired agent behaviour due to reward hacking
\[[26](#ref-pan_effects_2022), [34](#ref-skalse_defining_2022)\].

## Why we want to solve it

Addressing the issue of reward misspecification is important because it
is one of the many limiting factors that make RL difficult to apply.
Furthermore, due to reward hacking, the issue can lead to undesired
behaviour. The negative impacts of misbehaviour can be as simple as a
model underperforming in production and as dire as causing safety
concerns \[[16](#ref-hendrycks_unsolved_2022)\].

## Current solutions and their shortcomings

### Inverse Reinforcement Learning

Inverse Reinforcement Learning (IRL) \[[17](#ref-ho_generative_2016),
[25](#ref-ng_algorithms_2000), [43](#ref-ziebart_maximum_2008)\] is the
problem of extracting a reward function given observed expert behaviour
(demonstrations). While promising and perhaps suitable for many
problems, IRL has some limitations. For instance expert demonstrations
are not always available and can be difficult to obtain. Furthermore,
for many environments it is very difficult to determine the reward
function from the demonstrations \[[2](#ref-amin_towards_2016),
[7](#ref-choi_inverse_2011)\]. Another limitation is that model
performance may be limited to the performance of the experts from which
it is learning \[[11](#ref-evans_learning_2015-1),
[12](#ref-evans_learning_2015)\]. IRL is often also criticised for
overlooking side-effects \[[22](#ref-krakovna_penalizing_2019)\] and
encouraging power-seeking \[[36](#ref-turner_optimal_2021)\]. Even if
these issues were addressed, IRL does not necessarily address the
overarching problem, as reward hacking has been observed in the IRL
context as well \[[19](#ref-ibarz_reward_2018)\].

In general, IRL is considered to be a subfield of ***imitation
learning*** \[[1](#ref-abbeel_apprenticeship_2004)\], where the goal is
now to predict trajectories, given expert demonstrations. Imitation
learning faces similar limitations to those of IRL.

### Preference-based learning

Preference-based learning circumvents the need for demonstrations by
using a more direct signal of human preferences. This includes, for
example, directly asking users what they want via e.g. pairwise
comparisons \[[4](#ref-biyik_batch_2018),
[8](#ref-christiano_deep_2017), [30](#ref-sadigh_active_2017)\]. The
main approach is to express preferences via pairwise comparison. This
however can be limited in expressivity: consider a case in which two
sub-optimal but complimentary correct trajectories are presented. Under
pairwise comparison, there is no way to express the necessary
granularity in preferences for this example. Another potential issue is
that the expressed preferences may be different from the real
preferences.

## Proposed approach

Using advances in natural language processing, particularly in large
language models (LLMs) \[[6](#ref-brown_language_2020),
[31](#ref-sanh_multitask_2022), [37](#ref-vaswani_attention_2017)\] and
prompting techniques \[[14](#ref-gal_image_2022),
[28](#ref-reynolds_prompt_2021), [38](#ref-wu_ai_2022)\], and inspired
by their applications beyond a pure NLP context
\[[10](#ref-dosovitskiy_image_2021),
[27](#ref-ramesh_hierarchical_2022),
[29](#ref-rombach_high-resolution_2022)\], we can develop a more natural
interface between human and machine to specify goals and or rewards.
This is after-all how humans communicate desired outcomes to each other.
There already exist many recent works leveraging the expressivity of
language models in an RL context \[[5](#ref-brooks_-context_2022),
[9](#ref-ding_robot_2022), [15](#ref-gramopadhye_generating_2022),
[18](#ref-huang_inner_2022), [20](#ref-jiang_vima_2022),
[24](#ref-lu_neuro-symbolic_2022), [32](#ref-shridhar_cliport_2021),
[33](#ref-shridhar_perceiver-actor_2022),
[35](#ref-sumers_learning_2021), [39](#ref-yao_react_2022),
[40](#ref-yu_using_2022), [42](#ref-zhou_inverse_2020)\]. A number of
NL-RL-hybrid environments and datasets
\[[3](#ref-anderson_vision-and-language_2018),
[13](#ref-fan_minedojo_2022), [21](#ref-jiang_yunfan_vima_2022),
[23](#ref-liu_reinforcement_2018), [41](#ref-zholus_iglu_2022)\] have
accompanied many of these papers in the field. These works however
mostly focus on their contributions to planning performance, learning
efficiency and other more common RL metrics of success. Using techniques
similar to those developed by \[[26](#ref-pan_effects_2022)\] and taking
inspiration from the recent works cited above, this work hopes to
explore the question: **to what extent can natural language interfaces
curtail the issue of reward hacking in RL?**

# References

<span class="csl-left-margin">\[1\] </span><span
class="csl-right-inline">Abbeel, P. and Ng, A.Y. 2004. [Apprenticeship
learning via inverse reinforcement
learning](https://doi.org/10.1145/1015330.1015430). *Proceedings of the
twenty-first international conference on Machine learning* (New York,
NY, USA, Jul. 2004), 1.</span>

<span class="csl-left-margin">\[2\] </span><span
class="csl-right-inline">Amin, K. and Singh, S. 2016. [Towards Resolving
Unidentifiability in Inverse Reinforcement
Learning](https://doi.org/10.48550/arXiv.1601.06569). arXiv.</span>

<span class="csl-left-margin">\[3\] </span><span
class="csl-right-inline">Anderson, P. et al. 2018. [Vision-and-Language
Navigation: Interpreting Visually-Grounded Navigation Instructions in
Real
Environments](https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html).
(2018), 3674–3683.</span>

<span class="csl-left-margin">\[4\] </span><span
class="csl-right-inline">Biyik, E. and Sadigh, D. 2018. [Batch Active
Preference-Based Learning of Reward
Functions](https://proceedings.mlr.press/v87/biyik18a.html).
*Proceedings of The 2nd Conference on Robot Learning* (Oct. 2018),
519–528.</span>

<span class="csl-left-margin">\[5\] </span><span
class="csl-right-inline">Brooks, E. et al. 2022. [In-Context Policy
Iteration](https://doi.org/10.48550/arXiv.2210.03821). arXiv.</span>

<span class="csl-left-margin">\[6\] </span><span
class="csl-right-inline">Brown, T.B. et al. 2020. [Language Models are
Few-Shot Learners](http://arxiv.org/abs/2005.14165).</span>

<span class="csl-left-margin">\[7\] </span><span
class="csl-right-inline">Choi, J. and Kim, K.-E. 2011. [Inverse
Reinforcement Learning in Partially Observable
Environments](http://jmlr.org/papers/v12/choi11a.html). *Journal of
Machine Learning Research*. 12, 21 (2011), 691–730.</span>

<span class="csl-left-margin">\[8\] </span><span
class="csl-right-inline">Christiano, P. et al. 2017. [Deep reinforcement
learning from human
preferences](https://doi.org/10.48550/arXiv.1706.03741). arXiv.</span>

<span class="csl-left-margin">\[9\] </span><span
class="csl-right-inline">Ding, Y. et al. 2022. [Robot Task Planning and
Situation Handling in Open
Worlds](https://doi.org/10.48550/arXiv.2210.01287). arXiv.</span>

<span class="csl-left-margin">\[10\] </span><span
class="csl-right-inline">Dosovitskiy, A. et al. 2021. [An Image is Worth
16x16 Words: Transformers for Image Recognition at
Scale](https://doi.org/10.48550/arXiv.2010.11929). arXiv.</span>

<span class="csl-left-margin">\[11\] </span><span
class="csl-right-inline">Evans, O. et al. 2015. Learning the preferences
of bounded agents. (2015).</span>

<span class="csl-left-margin">\[12\] </span><span
class="csl-right-inline">Evans, O. et al. 2015. [Learning the
Preferences of Ignorant, Inconsistent
Agents](https://doi.org/10.48550/arXiv.1512.05832). arXiv.</span>

<span class="csl-left-margin">\[13\] </span><span
class="csl-right-inline">Fan, L. et al. 2022. [MineDojo: Building
Open-Ended Embodied Agents with Internet-Scale
Knowledge](https://doi.org/10.48550/arXiv.2206.08853). arXiv.</span>

<span class="csl-left-margin">\[14\] </span><span
class="csl-right-inline">Gal, R. et al. 2022. [An Image is Worth One
Word: <span class="nocase">Personalizing Text-to-Image Generation</span>
using Textual Inversion](https://doi.org/10.48550/arXiv.2208.01618).
arXiv.</span>

<span class="csl-left-margin">\[15\] </span><span
class="csl-right-inline">Gramopadhye, M. and Szafir, D. 2022.
[Generating Executable Action Plans with Environmentally-Aware Language
Models](https://doi.org/10.48550/arXiv.2210.04964). arXiv.</span>

<span class="csl-left-margin">\[16\] </span><span
class="csl-right-inline">Hendrycks, D. et al. 2022. [Unsolved Problems
in ML Safety](https://doi.org/10.48550/arXiv.2109.13916). arXiv.</span>

<span class="csl-left-margin">\[17\] </span><span
class="csl-right-inline">Ho, J. and Ermon, S. 2016. [Generative
Adversarial Imitation
Learning](https://proceedings.neurips.cc/paper/2016/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html).
*Advances in Neural Information Processing Systems* (2016).</span>

<span class="csl-left-margin">\[18\] </span><span
class="csl-right-inline">Huang, W. et al. 2022. [Inner Monologue:
Embodied Reasoning through Planning with Language
Models](https://doi.org/10.48550/arXiv.2207.05608). arXiv.</span>

<span class="csl-left-margin">\[19\] </span><span
class="csl-right-inline">Ibarz, B. et al. 2018. [Reward learning from
human preferences and demonstrations in
Atari](https://doi.org/10.48550/arXiv.1811.06521). arXiv.</span>

<span class="csl-left-margin">\[20\] </span><span
class="csl-right-inline">Jiang, Y. et al. 2022. [VIMA: General Robot
Manipulation with Multimodal
Prompts](https://doi.org/10.48550/arXiv.2210.03094). arXiv.</span>

<span class="csl-left-margin">\[21\] </span><span
class="csl-right-inline">Jiang, Y. et al. 2022. [VIMA: General Robot
Manipulation with Multimodal
Prompts](https://doi.org/10.5281/ZENODO.7127587). Zenodo.</span>

<span class="csl-left-margin">\[22\] </span><span
class="csl-right-inline">Krakovna, V. et al. 2019. [Penalizing side
effects using stepwise relative
reachability](https://doi.org/10.48550/arXiv.1806.01186). arXiv.</span>

<span class="csl-left-margin">\[23\] </span><span
class="csl-right-inline">Liu, E.Z. et al. 2018. [Reinforcement Learning
on Web Interfaces Using Workflow-Guided
Exploration](https://doi.org/10.48550/arXiv.1802.08802). arXiv.</span>

<span class="csl-left-margin">\[24\] </span><span
class="csl-right-inline">Lu, Y. et al. 2022. [Neuro-Symbolic Procedural
Planning with Commonsense
Prompting](https://doi.org/10.48550/arXiv.2206.02928). arXiv.</span>

<span class="csl-left-margin">\[25\] </span><span
class="csl-right-inline">Ng, A.Y. and Russell, S. 2000. Algorithms for
Inverse Reinforcement Learning. *In Proc. 17th International Conf. On
Machine Learning* (2000), 663–670.</span>

<span class="csl-left-margin">\[26\] </span><span
class="csl-right-inline">Pan, A. et al. 2022. [The Effects of Reward
Misspecification: Mapping and Mitigating Misaligned
Models](https://doi.org/10.48550/arXiv.2201.03544). arXiv.</span>

<span class="csl-left-margin">\[27\] </span><span
class="csl-right-inline">Ramesh, A. et al. 2022. [Hierarchical
Text-Conditional Image Generation with CLIP
Latents](https://doi.org/10.48550/arXiv.2204.06125). arXiv.</span>

<span class="csl-left-margin">\[28\] </span><span
class="csl-right-inline">Reynolds, L. and McDonell, K. 2021. [Prompt
Programming for Large Language Models: Beyond the Few-Shot
Paradigm](https://doi.org/10.48550/arXiv.2102.07350). arXiv.</span>

<span class="csl-left-margin">\[29\] </span><span
class="csl-right-inline">Rombach, R. et al. 2022. [High-Resolution Image
Synthesis With Latent Diffusion
Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html).
(2022), 10684–10695.</span>

<span class="csl-left-margin">\[30\] </span><span
class="csl-right-inline">Sadigh, D. et al. 2017. *[Active
preference-based learning of reward
functions](https://doi.org/10.15607/rss.2017.xiii.053)*.</span>

<span class="csl-left-margin">\[31\] </span><span
class="csl-right-inline">Sanh, V. et al. 2022. [Multitask Prompted
Training Enables Zero-Shot Task
Generalization](http://arxiv.org/abs/2110.08207).</span>

<span class="csl-left-margin">\[32\] </span><span
class="csl-right-inline">Shridhar, M. et al. 2021. [CLIPort: What and
Where Pathways for Robotic
Manipulation](https://doi.org/10.48550/arXiv.2109.12098). arXiv.</span>

<span class="csl-left-margin">\[33\] </span><span
class="csl-right-inline">Shridhar, M. et al. 2022. [Perceiver-Actor: A
Multi-Task Transformer for Robotic
Manipulation](https://doi.org/10.48550/arXiv.2209.05451). arXiv.</span>

<span class="csl-left-margin">\[34\] </span><span
class="csl-right-inline">Skalse, J. et al. 2022. [Defining and
Characterizing Reward
Hacking](https://doi.org/10.48550/arXiv.2209.13085). arXiv.</span>

<span class="csl-left-margin">\[35\] </span><span
class="csl-right-inline">Sumers, T.R. et al. 2021. [Learning Rewards
from Linguistic Feedback](https://doi.org/10.48550/arXiv.2009.14715).
arXiv.</span>

<span class="csl-left-margin">\[36\] </span><span
class="csl-right-inline">Turner, A.M. et al. 2021. [Optimal Policies
Tend to Seek Power](https://doi.org/10.48550/arXiv.1912.01683).
arXiv.</span>

<span class="csl-left-margin">\[37\] </span><span
class="csl-right-inline">Vaswani, A. et al. 2017. [Attention Is All You
Need](http://arxiv.org/abs/1706.03762).</span>

<span class="csl-left-margin">\[38\] </span><span
class="csl-right-inline">Wu, T. et al. 2022. [AI Chains: Transparent and
Controllable Human-AI Interaction by Chaining Large Language Model
Prompts](https://doi.org/10.1145/3491102.3517582). *Proceedings of the
2022 CHI Conference on Human Factors in Computing Systems* (New York,
NY, USA, Apr. 2022), 1–22.</span>

<span class="csl-left-margin">\[39\] </span><span
class="csl-right-inline">Yao, S. et al. 2022. [ReAct: Synergizing
Reasoning and Acting in Language
Models](https://doi.org/10.48550/arXiv.2210.03629). arXiv.</span>

<span class="csl-left-margin">\[40\] </span><span
class="csl-right-inline">Yu, A. and Mooney, R.J. 2022. [Using Both
Demonstrations and Language Instructions to Efficiently Learn Robotic
Tasks](https://doi.org/10.48550/arXiv.2210.04476). arXiv.</span>

<span class="csl-left-margin">\[41\] </span><span
class="csl-right-inline">Zholus, A. et al. 2022. [IGLU Gridworld: Simple
and Fast Environment for Embodied Dialog
Agents](https://doi.org/10.48550/arXiv.2206.00142). arXiv.</span>

<span class="csl-left-margin">\[42\] </span><span
class="csl-right-inline">Zhou, L. and Small, K. 2020. [Inverse
Reinforcement Learning with Natural Language
Goals](https://doi.org/10.48550/arXiv.2008.06924). arXiv.</span>

<span class="csl-left-margin">\[43\] </span><span
class="csl-right-inline">Ziebart, B.D. et al. 2008. Maximum entropy
inverse reinforcement learning. *Proceedings of the 23rd national
conference on Artificial intelligence - Volume 3* (Chicago, Illinois,
Jul. 2008), 1433–1438.</span>
