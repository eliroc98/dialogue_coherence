# Approaching dialog coherence with multiple fine-tuned BERT models
The aim of this project is to approach the problem of evaluating a dialogue's coherence by exploiting different measures, which are related to different dialogue aspects, such as logical coherence, debaters' intentions, emotions and discussed topics.

## Related work

### Coherence evaluation
In the following there is a list of recent papers regarding novel coherence evaluation metrics and methods. One important thing to take into account, in all these works, is the evaluation procedure they make to test their metrics.

[Ye et al., “Towards Quantifiable Dialogue Coherence Evaluation.”](https://aclanthology.org/2021.acl-long.211)[^ye2021] proposed Quantifiable Dialogue Coherence Evaluation (namely QuantiDCE), which is a coherence measure aimig at having a high correlations with human evaluations in an automatic fashion. The main features in QuantiDCE are:
- it models the task in a multi-level setting which is closer to the actual human rating, instead of simplifying the coherence evaluation task solving it in a two-level setting (i.e., coherent or incoherent). Indeed, humans usually adopt Likert scaling and give coherence scores from multiple levels like 1 to 5;
- it can be fine-tuned with a small set of actual human-annotated coherence scores.
👩‍💻 Links: [GitHub](https://github.com/James-Yip/QuantiDCE) with the checkpoint.

[Pang et al., “Towards Holistic and Automatic Evaluation of Open-Domain Dialogue Generation.”](https://aclanthology.org/2020.acl-main.333)[^pang2020] proposed holistic evaluation metrics that capture different aspects of open-domain dialogues. Their metrics consists of 1) GPT-2 based context coherence between sentences in a dialogue, (2) GPT-2 based fluency in phrasing, (3) n-gram based diversity in responses to augmented queries, and (4) textual-entailment-inference based logical self-consistency. Their metrics strongly correlate with human evaluations. The main features (for this project) in this work are:
- the way they measure context coherence using a GPT-2 fine-tuned on the next sentence prediciton task on the dialogue dataset of interest
- the way they measure the logical self-consistency using a pretrained Multi-Genre Natural Language Inference[^williams2018] model to label if the relation of the response and the utterance history of the same agent is logically consistent;
- they evaluate the relation across the metrics.
👩‍💻 Links: [GitHub](https://github.com/alexzhou907/dialogue_evaluation) with GPT-2 trained on DailyDialog.

These metrics achieve a better correlation with human annotations, fixing the problem of other metrics presented in the past which do not use semantic information, but they are still relevant in many works. Such non-semantic metrics are:
- BLEU[^papineni2002]. The aim of BLEU is to evaluate traslations. The primary programming task for a BLEU implementor is to compare n-grams of the candidate with the n-grams of the reference translation and count the number of matches. These matches are position independent. The more the matches, the better the candidate translation is.
- METEOR[^banerjee2005]. It is an automatic metric for machine translation evaluation that is based on a generalized concept of unigram matching between the machineproduced translation and human-produced reference translations.
- ROUGE[^lin2004]. ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation. It includes measures to automatically determine the quality of a summary by comparing it to other (ideal) summaries created by humans. The measures count the number of overlapping units such as n-gram, word sequences, and word pairs between the computer-generated summary to be evaluated and the ideal summaries created by humans.

[Liu et al., “How NOT To Evaluate Your Dialogue System.”](https://aclanthology.org/D16-1230)[^liu2016] and [Novikova et al., “Why We Need New Evaluation Metrics for NLG.”](https://aclanthology.org/D17-1238)[^novikova2017] demonstrate that these matrics correlate poorly with human judgments due to the absence of semantic information.

Thus, one of the first improvement to be applied in coherence metrics consists in leveraging on semantic information to try to better correlate with human judgements. The following metrics are proposed to integrate semantic information to enrich the evaluation, and some of them are classified as learnable metrics:
- BERTScore[^zhang2020]. It computes a similarity score for each token in the candidate sentence with each token in the reference sentence. However, instead of exact matches, it computes token similarity using contextual embeddings. This paper also provides a good summary of the evolution concerning automatic evaluation of natural language generation. 👩‍💻 Links: [GitHub](https://github.com/Tiiiger/bert_score).
- ADEM[^lowe2017]. The authors propose an automatic dialogue evaluation as a learning problem. We present an evaluation model (ADEM) that learns to predict human-like scores to input responses, using a new dataset of human response scores. They use RNN.
- RUBER[^tao2018]. In this paper, the authors propose RUBER, a Referenced metric (an embedding-based scorer measures the similarity between a generated reply and the groundtruth) and Unreferenced metric (a neural network-based scorer measures the relatedness between the generated reply and its query) Blended Evaluation Routine, which evaluates a reply by taking into consideration both a groundtruth reply and a query (previous user-issued utterance). Our metric is learnable, but its training does not require labels of human satisfaction. No BERT.
- BERT-RUBER[^ghazarian2019]. It applies contextualized word embeddings to automatic evaluation of open-domain dialogue systems. The experiments showed that the unreferenced scores of RUBER metric can be improved by considering contextualized word embeddings which include richer representations of words and their context.
- BLEURT[^sellam2020]. It is a learned evaluation metric based on BERT that can model human judgments with a few thousand possibly biased training examples. A key aspect of their approach is a novel pre-training scheme that uses millions of synthetic examples to help the model generalize.
👩‍💻 Links: [GitHub](https://github.com/google-research/bleurt) with checkpoint.
- **GRADE[^huang2020]**. The authors first consider that the graph structure constituted with topics in a dialogue can accurately depict the underlying communication logic, which is a more natural way to produce persuasive metrics. Capitalized on the topic-level dialogue graph, the authors propose a new evaluation metric GRADE, which stands for Graph-enhanced Representations for Automatic Dialogue Evaluation. Specifically, GRADE incorporates both coarsegrained utterance-level contextualized representations and fine-grained topic-level graph representations to evaluate dialogue coherence. The graph representations are obtained by reasoning over topic-level dialogue graphs enhanced with the evidence from a commonsense graph, including k-hop neighboring representations and hop-attention weights. They use DailyDialog to train their model. 👩‍💻 Links: [GitHub](https://github.com/li3cmz/GRADE), it also has a model checkpoint!

### Structure of attention in transformers
[Clark et al., “What Does BERT Look At?”](http://arxiv.org/abs/1906.04341)[^clark2019] propose methods for analysing the attention mechanisms of pre-trained models and apply them to BERT. BERT’s attention heads exhibit patterns such as attending to delimiter tokens, specific positional offsets, or broadly attending over the whole sentence, with heads in the same layer often exhibiting similar behaviors. We further show that certain attention heads correspond well to linguistic notions of syntax and coreference. They propose an attention-based probing classifier and use it to further demonstrate that substantial syntactic information is captured in BERT’s attention. What is interesting for this work are many aspects of their findings:
- BERT is capable of learning a lot of linguistic knowledge in its attention maps. This is relevant because the behaviour of the attention heads emerges purely from self-supervised training on unlabeled data, without explicit supervision for syntax or coreference;
- they not only highlight the patterns found in attention maps, but they also validate this qualitative finding investigating individual attention heads to probe what aspects of language they have learned. They evaluate attention heads on labeled datasets for task like dependency parsing and coreference resolution;
- they cluster attention heads to show that heads tend to cluster depending on their behaviour and their layer.
👩‍💻 Links: [GitHub](https://github.com/li3cmz/GRADE) code to extract attention masks!

[Raganato and Tiedemann, “An Analysis of Encoder Representations in Transformer-Based Machine Translation.”](https://aclanthology.org/W18-5431)[^raganato2018] found that attention encapsulate dependency relations and syntactic and semantic behavior across layers.

[Manning et al., “Emergent Linguistic Structure in Artificial Neural Networks Trained by Self-Supervision.”](https://www.pnas.org/doi/full/10.1073/pnas.1907367117)[^manning2020] describes from scratch the problem of analysing linguistic structure in attention masks. Along with explaining all the building bricks to understand the problem itself, they explain also how to evaluate the attention head in representing a certain attention pattern. In particular, they use attention heads as simple classifiers, examining the most-attended-to word at each position. They evaluate whether the attention head is expressing a particular linguistic relationship by computing how often the most-attendend-to word is in that relationship with the input word. They compute the precision expressing the head's capability of identifying certain linguistic relationships: this score can be viewed as evaluating the attention head as a simple classifier that predicts the presence of the linguistic relationship of interest.

[Vig and Belinkov, “Analyzing the Structure of Attention in a Transformer Language Model.”](https://aclanthology.org/W19-4808)[^vig2019a] found that many attention heads specialize in particular part-of-speech tags and that different tags are targeted at different layer depths. They also found that the deepest layers capture the most distant relationships, and that attention aligns most strongly  with dependency relations in the middle layers where attention distance is lowest. Lastly, they suggest that the structure of attention is closely tied to the training objective.

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

#### Other datasets to analyse
- [DailyDialog++](https://github.com/iitmnlp/Dialogue-Evaluation-with-BERT)[^sai2020]. It allows for better training and robust evaluation of model-based metrics, they introduce the DailyDialog++ dataset, consisting of (i) five relevant responses for each context and (ii) five adversarially crafted irrelevant responses for each context. This dataset can be used to train a model classifying how coherent is a dialog in logical terms.
- DailyDialogEVAL[^huang2020], which is a subset of the adopted evaluation benchmark (Huang et al., 2020)[^huang2020], with 300 human rating data in total, and randomly split the data into training (90%) and validation (10%) sets.

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

This idea follows a procedure similar to what is proposed in [Raganato and Tiedemann, “An Analysis of Encoder Representations in Transformer-Based Machine Translation.”](https://aclanthology.org/W18-5431)[^raganato2018] and [Vig and Belinkov, “Analyzing the Structure of Attention in a Transformer Language Model.”](https://aclanthology.org/W19-4808)[^vig2019a]. The focus in this part of the work is to analyse the structure of the attention mask in BERT.

![attention_BERT](https://miro.medium.com/max/1400/0*AovFiJtn-LV-q2ey.gif)

Given the promising results obtained by the papers presented in the "Related Work" section, it should be interesting to leverage on attention masks to observe how the transformer works internally and the patterns it uses when addressing some specific tasks. Analysing attention masks has a twofold benefit:
1. it makes the coherence "interpretable", especially when using the encoder-decoder attention mask;
2. it offers a new way of making the coherence metric more verifiable. Indeed, if the pattern in attention masks observed for a new dialog is similar to those patterns observed in the ground truth, it could be that this similarity correlates with an effective coherence measure.

Another curiosity that could be satisfied is to analyse how different fine-tuning procedures for different tasks affect the pattern observed in the attention masks. An interesting point would be to discover that, when performing the emotion classification, the attention mask finds correspondences in the two-slot inputs which are semantically correlated with something "emotional".

[^li2017]: 
    Li, Yanran, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. “DailyDialog: A Manually Labelled Multi-Turn Dialogue Dataset.” In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 986–95. Taipei, Taiwan: Asian Federation of Natural Language Processing, 2017. https://aclanthology.org/I17-1099.

[^dziri2020]:
    Dziri, Nouha, Ehsan Kamalloo, Kory W. Mathewson, and Osmar Zaiane. “Evaluating Coherence in Dialogue Systems Using Entailment.” ArXiv:1904.03371 [Cs], March 31, 2020. http://arxiv.org/abs/1904.03371.

[^vig2019a]:
    Vig, Jesse, and Yonatan Belinkov. “Analyzing the Structure of Attention in a Transformer Language Model.” In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, 63–76. Florence, Italy: Association for Computational Linguistics, 2019. https://doi.org/10.18653/v1/W19-4808.

[^raganato2018]:
    Raganato, Alessandro, and Jörg Tiedemann. “An Analysis of Encoder Representations in Transformer-Based Machine Translation.” In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, 287–97. Brussels, Belgium: Association for Computational Linguistics, 2018. https://doi.org/10.18653/v1/W18-5431.

[^ye2021]:
    Ye, Zheng, Liucun Lu, Lishan Huang, Liang Lin, and Xiaodan Liang. “Towards Quantifiable Dialogue Coherence Evaluation.” In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2718–29. Online: Association for Computational Linguistics, 2021. https://doi.org/10.18653/v1/2021.acl-long.211.

[^pang2020]:
    Pang, Bo, Erik Nijkamp, Wenjuan Han, Linqi Zhou, Yixian Liu, and Kewei Tu. “Towards Holistic and Automatic Evaluation of Open-Domain Dialogue Generation.” In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 3619–29. Online: Association for Computational Linguistics, 2020. https://doi.org/10.18653/v1/2020.acl-main.333.

[^williams2018]:
    Williams, Adina, Nikita Nangia, and Samuel Bowman. “A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference.” In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), 1112–22. New Orleans, Louisiana: Association for Computational Linguistics, 2018. https://doi.org/10.18653/v1/N18-1101.

[^zhang2020]:
    Zhang*, Tianyi, Varsha Kishore*, Felix Wu*, Kilian Q. Weinberger, and Yoav Artzi. “BERTScore: Evaluating Text Generation with BERT,” 2019. https://openreview.net/forum?id=SkeHuCVFDr.

[^ghazarian2019]:
    Ghazarian, Sarik, Johnny Wei, Aram Galstyan, and Nanyun Peng. “Better Automatic Evaluation of Open-Domain Dialogue Systems with Contextualized Embeddings.” In Proceedings of the Workshop on Methods for Optimizing and Evaluating Neural Language Generation, 82–89. Minneapolis, Minnesota: Association for Computational Linguistics, 2019. https://doi.org/10.18653/v1/W19-2310.

[^papineni2002]:
    Papineni, Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu. “BLEU: A Method for Automatic Evaluation of Machine Translation.” In Proceedings of the 40th Annual Meeting on Association for Computational Linguistics, 311–18. ACL ’02. USA: Association for Computational Linguistics, 2002. https://doi.org/10.3115/1073083.1073135.

[^sellam2020]:
    Sellam, Thibault, Dipanjan Das, and Ankur Parikh. “BLEURT: Learning Robust Metrics for Text Generation.” In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 7881–92. Online: Association for Computational Linguistics, 2020. https://doi.org/10.18653/v1/2020.acl-main.704.

[^huang2020]:
    Huang, Lishan, Zheng Ye, Jinghui Qin, Liang Lin, and Xiaodan Liang. “GRADE: Automatic Graph-Enhanced Coherence Metric for Evaluating Open-Domain Dialogue Systems.” In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 9230–40. Online: Association for Computational Linguistics, 2020. https://doi.org/10.18653/v1/2020.emnlp-main.742.

[^banerjee2005]:
    Banerjee, Satanjeev, and Alon Lavie. “METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments.” In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, 65–72. Ann Arbor, Michigan: Association for Computational Linguistics, 2005. https://aclanthology.org/W05-0909.

[^lin2004]:
    Lin, Chin-Yew. “ROUGE: A Package for Automatic Evaluation of Summaries.” In Text Summarization Branches Out, 74–81. Barcelona, Spain: Association for Computational Linguistics, 2004. https://aclanthology.org/W04-1013.

[^lowe2017]:
    Lowe, Ryan, Michael Noseworthy, Iulian Vlad Serban, Nicolas Angelard-Gontier, Yoshua Bengio, and Joelle Pineau. “Towards an Automatic Turing Test: Learning to Evaluate Dialogue Responses.” In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 1116–26. Vancouver, Canada: Association for Computational Linguistics, 2017. https://doi.org/10.18653/v1/P17-1103.

[^tao2018]:
    Tao, Chongyang, Lili Mou, Dongyan Zhao, and Rui Yan. “RUBER: An Unsupervised Method for Automatic Evaluation of Open-Domain Dialog Systems.” Proceedings of the AAAI Conference on Artificial Intelligence 32, no. 1 (April 25, 2018). https://ojs.aaai.org/index.php/AAAI/article/view/11321.

[^liu2016]:
    Liu, Chia-Wei, Ryan Lowe, Iulian Serban, Mike Noseworthy, Laurent Charlin, and Joelle Pineau. “How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation.” In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, 2122–32. Austin, Texas: Association for Computational Linguistics, 2016. https://doi.org/10.18653/v1/D16-1230.

[^novikova2017]:
    Novikova, Jekaterina, Ondřej Dušek, Amanda Cercas Curry, and Verena Rieser. “Why We Need New Evaluation Metrics for NLG.” In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, 2241–52. Copenhagen, Denmark: Association for Computational Linguistics, 2017. https://doi.org/10.18653/v1/D17-1238.

[^sai2020]:
    Sai, Ananya B., Akash Kumar Mohankumar, Siddhartha Arora, and Mitesh M. Khapra. “Improving Dialog Evaluation with a Multi-Reference Adversarial Dataset and Large Scale Pretraining.” ArXiv:2009.11321 [Cs], September 23, 2020. http://arxiv.org/abs/2009.11321.

[^clark2019]:
    Clark, Kevin, Urvashi Khandelwal, Omer Levy, and Christopher D. Manning. “What Does BERT Look At? An Analysis of BERT’s Attention.” ArXiv:1906.04341 [Cs], June 10, 2019. http://arxiv.org/abs/1906.04341.

[^vig2019b]:
    Vig, Jesse. “A Multiscale Visualization of Attention in the Transformer Model.” In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, 37–42. Florence, Italy: Association for Computational Linguistics, 2019. https://doi.org/10.18653/v1/P19-3007.

[^clark2019]:
    Clark, Kevin, Urvashi Khandelwal, Omer Levy, and Christopher D. Manning. “What Does BERT Look At? An Analysis of BERT’s Attention.” ArXiv:1906.04341 [Cs], June 10, 2019. http://arxiv.org/abs/1906.04341.

[^manning2020]:
    Manning, Christopher D., Kevin Clark, John Hewitt, Urvashi Khandelwal, and Omer Levy. “Emergent Linguistic Structure in Artificial Neural Networks Trained by Self-Supervision.” Proceedings of the National Academy of Sciences 117, no. 48 (December 2020): 30046–54. https://doi.org/10.1073/pnas.1907367117.

