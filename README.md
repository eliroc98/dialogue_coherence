# Approaching dialog coherence with multiple fine-tuned BERT
The aim of this project is to approach the problem of evaluating a dialogue's coherence by exploiting different measures, which are related to different dialogue aspects, such as logical coherence, debaters' intentions, emotions and discussed topics.

## Methodology
### Data
DailyDialog [Li et al., “DailyDialog.”](https://aclanthology.org/I17-1099) [data](https://aclanthology.org/attachments/I17-1099.Datasets.zip) is a dataset containing more than 13000 dialogues having
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
The selected dataset does not necessarily require many preparations steps. One major task to complete to have the final dataset is to add one feature to address the problem of measuring the logical coherence. To do so, it is possible to follow a similar procedure to what is suggested in [Dziri et al., “Evaluating Coherence in Dialogue Systems Using Entailment.”](http://arxiv.org/abs/1904.03371):
- assigning the label "*coherent*" to those sequences of utterances which appear in a dialogue from DailyDialog;
- assigning the label "*not coherent*" to sequences of utterances belonging to perturbations of dialogues obtained by the perturbations of genuine ones.

### Modeling strategy
DailyDialog is used as input data in 4 BERT pre-trained models to perform the fine-tuning procedure. The objective here is to have 4 specilized models to capture the different aspects defined before: logical coherence, debaters' intentions, emotions and discussed topics.

Each dialogue is processed by each BERT model in such a way that many pairs of sequences of utterances are fed into BERT respecting the order in which they appear in the dialogue. The input can be viewed as a pair: the first slot contains one or more utterances representing the dialogue history; the second slot contains one utterance, representing the next utterance in the dialogue. Depending on the assigned task, BERT produces a different outcome:
- logical coherence - BERT outputs a value representing the probability of the second utterance-slot to be logically coherent with the first utterance-slot;
- intention - BERT outputs a value representing the probability of the second utterance-slot to be "intentionally" coherent with the first utterance-slot. "Intentionally coherent" means that the pattern observed aggregating the intentions of utterances in the dialogue history allows for a certain type of intention to be the next utterance intention;
- emotion - BERT outputs a value representing the probability of the second utterance-slot to be "emotionally" coherent with the first utterance-slot. "Emotionally coherent" means that the pattern observed aggregating the emotions of utterances in the dialogue history allows for a certain type of emotion to be the next utterance emotion;
- topic - BERT outputs a value representing the probability of the second utterance-slot to be "topic" coherent with the first utterance-slot. "Topic coherent" means that the utterances are about the same topic: the dataset does not allow for multiple topics in the same dialogue, thus it is not be possible to evaluate an in-dialogue topic change.

Once 