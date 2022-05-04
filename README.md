# Approaching dialog coherence with multiple fine-tuned BERT models
The aim of this project is to approach the problem of evaluating a dialogue's coherence by exploiting different measures, which are related to different dialogue aspects, such as logical coherence, debaters' intentions, emotions and discussed topics.

## Related work
In the following there is a list of recent papers regarding novel coherence evaluation metrics and methods. One important thing to take into account, in all these works, is the evaluation procedure they make to test their metrics.


[Ye et al., “Towards Quantifiable Dialogue Coherence Evaluation.”](https://aclanthology.org/2021.acl-long.211)[^ye2021] proposed Quantifiable Dialogue Coherence Evaluation (namely QuantiDCE), which is a coherence measure aimig at having a high correlations with human evaluations in an automatic fashion. The main features in QuantiDCE are:
- it models the task in a multi-level setting which is closer to the actual human rating, instead of simplifying the coherence evaluation task solving it in a two-level setting (i.e., coherent or incoherent). Indeed, humans usually adopt Likert scaling and give coherence scores from multiple levels like 1 to 5;
- it can be fine-tuned with a small set of actual human-annotated coherence scores.

[Pang et al., “Towards Holistic and Automatic Evaluation of Open-Domain Dialogue Generation.”](https://aclanthology.org/2020.acl-main.333)[^pang2020] proposed holistic evaluation metrics that capture different aspects of open-domain dialogues. Their metrics consists of 1) GPT-2 based context coherence between sentences in a dialogue, (2) GPT-2 based fluency in phrasing, (3) n-gram based diversity in responses to augmented queries, and (4) textual-entailment-inference based logical self-consistency. Their metrics strongly correlate with human evaluations. The main features (for this project) in this work are:
- the way they measure context coherence using a GPT-2 fine-tuned on the next sentence prediciton task on the dialogue dataset of interest
- the way they measure the logical self-consistency using a pretrained Multi-Genre Natural Language Inference[^williams2018] model to label if the relation of the response and the utterance history of the same agent is logically consistent;
- they evaluate the relation across the metrics.

These two papers share some literature, which is reported in the following:
...


Concerning the structure of attention in transformers, 
...

## Methodology
### Data
The work presented in [Li et al., “DailyDialog.”](https://aclanthology.org/I17-1099)[^li2017] consists of a [dataset](https://aclanthology.org/attachments/I17-1099.Datasets.zip) containing more than 13000 dialogues having
- 7.9 (on average) speaker turns per dialogue;
- 114.7 (on average) tokens per dialogue;
- 14.6 (on average) tokens per utterance.

The peculiarity defining DailyDialog is the precise annotation describing 3 different dialogue aspects, namely
- emotions, specified for each utterance in the dialogue;
- intentions, specified for each utterance in the dialogue;
- topics, specified for each dialogue.

As a quick overview of the main DailyDialog's characteristics is listed in the following:
- **Daily Topics**: It covers ten categories ranging from ordinary life to financial topics, which is different from domain-specific datasets. 
- **Bi-turn Dialog Flow**: It conforms basic dialog act flows, such as Questions-Inform and Directives-Commissives bi-turn flows, making it different from question answering (QA) datasets and post-reply datasets. 
- **Certain Communication Pattern**: It follows unique multi-turn dialog flow patterns reflecting human communication style, which are rarely seen in task-oriented datasets.
- **Rich Emotion**: It contains rich emotions and is labeled manually to keep high-quality, which is distinguished from most existing dialog datasets.

#### Data preparation
The selected dataset does not necessarily require many preparations steps. One major task to complete to have the final dataset is to add many features to address the problem of measuring the logical coherence. To do so, it is possible to follow a similar procedure to what is suggested in [Dziri et al., “Evaluating Coherence in Dialogue Systems Using Entailment.”](http://arxiv.org/abs/1904.03371)[^dziri2020]:
- assigning the label "*logically-coherent*" to those sequences of utterances which appear in a dialogue from DailyDialog;
- assigning the label "*logically-not coherent*" to sequences of utterances belonging to perturbations of dialogues obtained by the perturbations of genuine ones.

### Modeling strategy
DailyDialog is used as input data in 4 BERT pre-trained models to perform the fine-tuning procedure. The objective here is to have 4 specilized models to capture the different aspects defined before: logical coherence, debaters' intentions, emotions and discussed topics.

Each dialogue is processed by each BERT model in such a way that many pairs of sequences of utterances are fed into BERT respecting the order in which they appear in the dialogue. The input can be viewed as a pair: the first slot contains one or more utterances representing the dialogue history; the second slot contains one utterance, representing the next utterance in the dialogue. Depending on the assigned task, BERT produces a different outcome:
- *logical coherence - BERT* outputs a value representing the probability of the second utterance-slot to be logically coherent with the first utterance-slot;
- *intention - BERT* outputs a probability vector representing the probabilities of the second utterance-slot to have a certain intention. "Intentionally coherent" then means that the pattern observed by aggregating the intentions of utterances in the dialogue history allows for a certain type of intention to be the next utterance intention;
- *emotion - BERT* outputs a probability vector representing the probabilities of the second utterance-slot to have a certain emotion. "Emotionally coherent" means that the pattern observed aggregating the emotions of utterances in the dialogue history allows for a certain type of emotion to be the next utterance emotion;
- *topic - BERT* outputs a probability vector representing the probabilities of the second utterance-slot to be about a certain topic. "Topic coherent" means that the utterances are about the same topic: the dataset does not allow for multiple topics in the same dialogue, thus it is not be possible to evaluate an in-dialogue topic change.

Once a dialogue is entirely processed, the probability patterns regarding logical coherence, intentions, emotions and topics are extracted and confronted with the patterns in the groud truth (consisting only of coherent dialogues). The degree of adherence of each pattern with the ground truth can be a proxy of the multi-dimension coherence, which is the objective of this work.

### Possible enhancement
One possible additional task based on the previously described methodology is to interpret which parts of two utterances in a dialogue are related according to the weights computed by BERT and stored in the encoder-decoder multi-head attention sub-layer, which performs an attention between the final encoder representation and the decoder representation, and in which each position of the decoder attends all positions in the last encoder layer.

This idea follows a procedure similar to what is proposed in [Raganato and Tiedemann, “An Analysis of Encoder Representations in Transformer-Based Machine Translation.”](https://aclanthology.org/W18-5431)[^raganato2018] and [Vig and Belinkov, “Analyzing the Structure of Attention in a Transformer Language Model.”](https://aclanthology.org/W19-4808)[^vig2019]. The focus in this part of the work is to analyse the structure of the attention mask in BERT: indeed,
- [Raganato and Tiedemann, “An Analysis of Encoder Representations in Transformer-Based Machine Translation.”](https://aclanthology.org/W18-5431)[^raganato2018] found that attention encapsulate dependency relations and syntactic and semantic behavior across layers;
- [Vig and Belinkov, “Analyzing the Structure of Attention in a Transformer Language Model.”](https://aclanthology.org/W19-4808)[^vig2019] found that many attention heads specialize in particular part-of-speech tags and that different tags are targeted at different layer depths. They also found that the deepest layers capture the most distant relationships, and that attention aligns most strongly  with dependency relations in the middle layers where attention distance is lowest. Lastly, they suggest that the structure of attention is closely tied to the training objective.

![attention_BERT](https://miro.medium.com/max/1400/0*AovFiJtn-LV-q2ey.gif)

Given the promising results obtained by the two presented papers, it should be interesting to leverage on attention masks to observe how the transformer works internally and the patterns it uses when addressing some specific tasks. Analysing attention masks has a twofold benefit:
1. it makes the coherence "interpretable", especially when using the encoder-decoder attention mask;
2. it offers a new way of making the coherence metric more verifiable. Indeed, if the pattern in attention masks observed for a new dialog is similar to those patterns observed in the ground truth, it could be that this similarity correlates with an effective coherence measure.

Another curiosity that could be satisfied is to analyse how different fine-tuning procedures for different tasks affect the pattern observed in the attention masks. An interesting point would be to discover that, when performing the emotion classification, the attention mask finds correspondences in the two-slot inputs which are semantically correlated with something "emotional".

[^li2017]: 
    Li, Yanran, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. “DailyDialog: A Manually Labelled Multi-Turn Dialogue Dataset.” In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 986–95. Taipei, Taiwan: Asian Federation of Natural Language Processing, 2017. https://aclanthology.org/I17-1099.

[^dziri2020]:
    Dziri, Nouha, Ehsan Kamalloo, Kory W. Mathewson, and Osmar Zaiane. “Evaluating Coherence in Dialogue Systems Using Entailment.” ArXiv:1904.03371 [Cs], March 31, 2020. http://arxiv.org/abs/1904.03371.

[^vig2019]:
    Vig, Jesse, and Yonatan Belinkov. “Analyzing the Structure of Attention in a Transformer Language Model.” In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, 63–76. Florence, Italy: Association for Computational Linguistics, 2019. https://doi.org/10.18653/v1/W19-4808.

[^raganato2018]:
    Raganato, Alessandro, and Jörg Tiedemann. “An Analysis of Encoder Representations in Transformer-Based Machine Translation.” In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, 287–97. Brussels, Belgium: Association for Computational Linguistics, 2018. https://doi.org/10.18653/v1/W18-5431.

[^ye2021]:
    Ye, Zheng, Liucun Lu, Lishan Huang, Liang Lin, and Xiaodan Liang. “Towards Quantifiable Dialogue Coherence Evaluation.” In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2718–29. Online: Association for Computational Linguistics, 2021. https://doi.org/10.18653/v1/2021.acl-long.211.

[^pang2020]:
    Pang, Bo, Erik Nijkamp, Wenjuan Han, Linqi Zhou, Yixian Liu, and Kewei Tu. “Towards Holistic and Automatic Evaluation of Open-Domain Dialogue Generation.” In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 3619–29. Online: Association for Computational Linguistics, 2020. https://doi.org/10.18653/v1/2020.acl-main.333.

[^williams2018]:
    Williams, Adina, Nikita Nangia, and Samuel Bowman. “A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference.” In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), 1112–22. New Orleans, Louisiana: Association for Computational Linguistics, 2018. https://doi.org/10.18653/v1/N18-1101.
