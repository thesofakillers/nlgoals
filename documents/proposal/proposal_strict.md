



## The Problem

In Reinforcement Learning (RL), the goal of a particular autonomous
agent is formalised in the form of a reward signal emitted by the
environment to the agent. This reward signal is typically computed via
some handcrafted reward function. However, handcrafted reward functions
can be difficult to specify for more complex problems and environments,
and can lead to undesired agent behaviour due to reward hacking
\[[14](#ref-pan_effects_2022)\].

## Why We Want To Solve It

Relying on handcrafted reward functions can be tedious, requiring at
times ample domain knowledge and mental effort. Furthermore, after
design, the reward function has to be manually implemented as part of
the agent’s environment. Finally, handcrafted reward functions may
suffer from bias and human error, leading to subpar or undesired
performance of our models. Generally, these are symptoms signaling
difficulty in scaling and generalisation. In the case of undesired model
performance, this has safety implications
\[[11](#ref-hendrycks_unsolved_2022)\].

## Current Solutions and their Shortcomings

### Inverse Reinforcement Learning

Inverse Reinforcement Learning (IRL) \[[12](#ref-ho_generative_2016),
[13](#ref-ng_algorithms_2000), [22](#ref-ziebart_maximum_2008)\] is the
problem of extracting a reward function given observed expert behaviour
(demonstrations). While promising and perhaps suitable for many
problems, IRL presents some limitations:

-   Expert demonstrations are not always available and can be difficult
    to obtain
-   For many environments it is very difficult to determine the reward
    function from the demonstrations.
    -   There is some research addressing this issue
        \[[2](#ref-amin_towards_2016), [5](#ref-choi_inverse_2011)\].
-   Model performance may be limited to the performance of the experts
    from which it is learning.
    -   There is some research addressing this issue
        \[[8](#ref-evans_learning_2015-1),
        [9](#ref-evans_learning_2015)\].
-   Natural intelligent agents (e.g. humans) don’t always need expert
    demonstrations to learn a reward function, so this is indicative of
    a lack of generalisation.
    <!-- - [Model Mis-specification and Inverse Reinforcement Learning | Academically Interesting (wordpress.com)](https://jsteinhardt.wordpress.com/2017/02/07/model-mis-specification-and-inverse-reinforcement-learning/) -->

IRL has a considerable overlap with ***imitation learning***
\[[1](#ref-abbeel_apprenticeship_2004)\], where the goal is now to
predict trajectories, given expert demonstrations. Imitation learning
faces similar limitations to those of IRL.

### Preference-Based Learning

Preference-based learning circumvents the need for demonstrations by
using a more direct signal of human preferences. This includes, for
example, directly asking users what they want via e.g. pairwise
comparisons \[[3](#ref-biyik_batch_2018),
[6](#ref-christiano_deep_2017), [18](#ref-sadigh_active_2017)\].

-   Expression of preferences via pairwise comparison can be limited.
-   Expressed preferences may be different from real preferences.

## Proposed Approach

Using advances in natural language processing, particularly in large
language models (LLMs) \[[4](#ref-brown_language_2020),
[19](#ref-sanh_multitask_2022), [20](#ref-vaswani_attention_2017)\] and
prompting techniques \[[10](#ref-gal_image_2022),
[16](#ref-reynolds_prompt_2021), [21](#ref-wu_ai_2022)\], and inspired
by their applications beyond a pure NLP context
\[[7](#ref-dosovitskiy_image_2021), [15](#ref-ramesh_hierarchical_2022),
[17](#ref-rombach_high-resolution_2022)\], we can develop a more natural
interface between human and machine to specify goals and or rewards.
This is after-all how humans communicate desired outcomes to each other.

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
class="csl-right-inline">Biyik, E. and Sadigh, D. 2018. [Batch Active
Preference-Based Learning of Reward
Functions](https://proceedings.mlr.press/v87/biyik18a.html).
*Proceedings of The 2nd Conference on Robot Learning* (Oct. 2018),
519–528.</span>

<span class="csl-left-margin">\[4\] </span><span
class="csl-right-inline">Brown, T.B. et al. 2020. [Language Models are
Few-Shot Learners](http://arxiv.org/abs/2005.14165).</span>

<span class="csl-left-margin">\[5\] </span><span
class="csl-right-inline">Choi, J. and Kim, K.-E. 2011. [Inverse
Reinforcement Learning in Partially Observable
Environments](http://jmlr.org/papers/v12/choi11a.html). *Journal of
Machine Learning Research*. 12, 21 (2011), 691–730.</span>

<span class="csl-left-margin">\[6\] </span><span
class="csl-right-inline">Christiano, P. et al. 2017. [Deep reinforcement
learning from human
preferences](https://doi.org/10.48550/arXiv.1706.03741). arXiv.</span>

<span class="csl-left-margin">\[7\] </span><span
class="csl-right-inline">Dosovitskiy, A. et al. 2021. [An Image is Worth
16x16 Words: Transformers for Image Recognition at
Scale](https://doi.org/10.48550/arXiv.2010.11929). arXiv.</span>

<span class="csl-left-margin">\[8\] </span><span
class="csl-right-inline">Evans, O. et al. 2015. Learning the preferences
of bounded agents. (2015).</span>

<span class="csl-left-margin">\[9\] </span><span
class="csl-right-inline">Evans, O. et al. 2015. [Learning the
Preferences of Ignorant, Inconsistent
Agents](https://doi.org/10.48550/arXiv.1512.05832). arXiv.</span>

<span class="csl-left-margin">\[10\] </span><span
class="csl-right-inline">Gal, R. et al. 2022. [An Image is Worth One
Word: <span class="nocase">Personalizing Text-to-Image Generation</span>
using Textual Inversion](https://doi.org/10.48550/arXiv.2208.01618).
arXiv.</span>

<span class="csl-left-margin">\[11\] </span><span
class="csl-right-inline">Hendrycks, D. et al. 2022. [Unsolved Problems
in ML Safety](https://doi.org/10.48550/arXiv.2109.13916). arXiv.</span>

<span class="csl-left-margin">\[12\] </span><span
class="csl-right-inline">Ho, J. and Ermon, S. 2016. [Generative
Adversarial Imitation
Learning](https://proceedings.neurips.cc/paper/2016/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html).
*Advances in Neural Information Processing Systems* (2016).</span>

<span class="csl-left-margin">\[13\] </span><span
class="csl-right-inline">Ng, A.Y. and Russell, S. 2000. Algorithms for
Inverse Reinforcement Learning. *In Proc. 17th International Conf. On
Machine Learning* (2000), 663–670.</span>

<span class="csl-left-margin">\[14\] </span><span
class="csl-right-inline">Pan, A. et al. 2022. [The Effects of Reward
Misspecification: Mapping and Mitigating Misaligned
Models](https://doi.org/10.48550/arXiv.2201.03544). arXiv.</span>

<span class="csl-left-margin">\[15\] </span><span
class="csl-right-inline">Ramesh, A. et al. 2022. [Hierarchical
Text-Conditional Image Generation with CLIP
Latents](https://doi.org/10.48550/arXiv.2204.06125). arXiv.</span>

<span class="csl-left-margin">\[16\] </span><span
class="csl-right-inline">Reynolds, L. and McDonell, K. 2021. [Prompt
Programming for Large Language Models: Beyond the Few-Shot
Paradigm](https://doi.org/10.48550/arXiv.2102.07350). arXiv.</span>

<span class="csl-left-margin">\[17\] </span><span
class="csl-right-inline">Rombach, R. et al. 2022. [High-Resolution Image
Synthesis With Latent Diffusion
Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html).
(2022), 10684–10695.</span>

<span class="csl-left-margin">\[18\] </span><span
class="csl-right-inline">Sadigh, D. et al. 2017. *[Active
preference-based learning of reward
functions](https://doi.org/10.15607/rss.2017.xiii.053)*.</span>

<span class="csl-left-margin">\[19\] </span><span
class="csl-right-inline">Sanh, V. et al. 2022. [Multitask Prompted
Training Enables Zero-Shot Task
Generalization](http://arxiv.org/abs/2110.08207).</span>

<span class="csl-left-margin">\[20\] </span><span
class="csl-right-inline">Vaswani, A. et al. 2017. [Attention Is All You
Need](http://arxiv.org/abs/1706.03762).</span>

<span class="csl-left-margin">\[21\] </span><span
class="csl-right-inline">Wu, T. et al. 2022. [AI Chains: Transparent and
Controllable Human-AI Interaction by Chaining Large Language Model
Prompts](https://doi.org/10.1145/3491102.3517582). *Proceedings of the
2022 CHI Conference on Human Factors in Computing Systems* (New York,
NY, USA, Apr. 2022), 1–22.</span>

<span class="csl-left-margin">\[22\] </span><span
class="csl-right-inline">Ziebart, B.D. et al. 2008. Maximum entropy
inverse reinforcement learning. *Proceedings of the 23rd national
conference on Artificial intelligence - Volume 3* (Chicago, Illinois,
Jul. 2008), 1433–1438.</span>
