# (Bi-)Weekly NLP Research  Paper Series

|   |Direct Submission  |ARR Commit |Author Response    |Notification   |Conference |Notice |
|---    |---    |---    |---    |---    |---    |---    |
|[SIGDIAL](https://2022.sigdial.org/call-for-papers/)   |05/11  |06/18  |-  |07/02  |09/07 - 09/09  |Edinburgh  |
|[COLING](https://coling2022.org/coling)    |05/17  |-  |-  |08/15  |10/12 - 10/15  |Gyeongju, Korea    |
|[EMNLP](https://2022.emnlp.org/calls/papers/Important-Dates)   |06/24  |07/24  |08/23 - 08/29  |10/06  |12/07 - 12/11  |Abu Dhabi, (ARR Withdraw: 05/24)   |
|[AACL](https://www.aacl2022.org/Submission/paper)  |07/15  |08/21  |08/15 - 08/21  |09/20  |11/21 - 11/24  |Taiwan |
|   |   |   |   |   |   |   |
|[ACL Rolling Review](https://aclrollingreview.org/six-week-cycles/)    |06/01, 07/15, 09/01, 10/15, 12/01, 01/15/2023  |   |   |   |   |   |
|   |   |   |   |   |   |   |

(Conference deadlines: https://aideadlin.es/?sub=ML,CV,NLP,RO,SP or https://ccfddl.github.io/)

⭐ **Goals:**

* Primary for sharing knowledge across different domains and catching up on recent updates.
* Contents:
    * Mainly or only collect interesting papers.
    * Summarize the approaches and frameworks.
    * Write strengths and weaknesses, and share potential applications to other domains.
    * Highlight some exciting papers. Template :
        * Title: the paper title
        * Summary: strengths and weaknesses
        * Deserve to note: specific paragraphs or designs deserve to be noted or further reading

🤖 **Schedule:**

* This document will _keep updating and release_ (bi-)weekly every Friday.

❤️ **Welcome:**

* You are **_more than welcome_** to invite any people and edit **any** parts of the documents, including but **not limited to** _deleting, adding, and modifying_ any parts.

## 🚀 Week 03 05/09/2022

### Dialogue & Multi-modal 

Deepmind 

* (Important paper) [Flamingo: a Visual Language Model for Few-Shot Learning](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/tackling-multiple-tasks-with-a-single-visual-language-model/flamingo.pdf)

Microsoft

* Vision-Language-Audio: [i-code: an integrative and Composable Multimodal Learning Framework](https://arxiv.org/abs/2205.01818#:~:text=i%2DCode%3A%20An%20Integrative%20and%20Composable%20Multimodal%20Learning%20Framework,-Ziyi%20Yang%2C%20Yuwei&text=Human%20intelligence%20is%20multimodal%3B%20we,to%20one%20or%20two%20modalities.)

OpenAI

* [Dalle-2](https://openai.com/dall-e-2/)
* A paper from Google X: [Translation between Molecules and Natural Language](https://arxiv.org/abs/2204.11817#:~:text=Joint%20representations%20between%20images%20and,semantic%2Dlevel%20control%20of%20images.)

### Question Answering & Retrieval

1. **RETRO (Deepmind)**: Borgeaud, Sebastian, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche et al. "Improving language models by retrieving from trillions of tokens." *arXiv preprint arXiv:2112.04426* (2021). [[_pdf_](https://arxiv.org/pdf/2112.04426.pdf)]
2. 

## 🚀 Week 02 04/29/2022 

### Dialogue Related Papers

RL for Dialog (NAACL 2022) - BY [****Sergey Levine****](https://twitter.com/svlevine)

* [Context-Aware Language Modeling for Goal-Oriented Dialogue Systems](https://sea-snell.github.io/CALM_LM_site/)
* [CHAI: A CHatbot AI for Task-Oriented Dialogue with Offline Reinforcement Learning](https://siddharthverma314.github.io/research/chai-acl-2022/)

Seeker and it relevant papers

* [Language Models that Seek for Knowledge: Modular Search & Generation for Dialogue and Prompt Completion](https://arxiv.org/abs/2203.13224) 
    * It outperforms GPT-3 regarding hallucination, and it is better than Blenderbot 2.0 
    * The paper is partially motivated by [Reason first, then respond: Modular Generation for Knowledge-infused Dialogue](https://arxiv.org/abs/2111.05204)and [Blenderbot 2.0](https://ai.facebook.com/blog/blender-bot-2-an-open-source-chatbot-that-builds-long-term-memory-and-searches-the-internet/). Many people may already know that Meta treats the task-oriented dialog (TOD) system [Cairaoke](https://ai.facebook.com/blog/project-cairaoke/) as one essential component of Metaverse, and Cairaoke integrates Blenderbot 2.0 to exhibit empathetic language and personality. In addition,  [Internet-Augmented Dialogue Generation](https://parl.ai/projects/sea/) is the code paper for Blenderbot 2.0, and I personally treat it as one excellent paper of that year. 
    * Deserve to note:
        * It integrates a search engine into the open-domain dialogue generation. The search engine firsts search the Internet to retrieve documents and keep the Top 5. Then a knowledge module to select more relevant knowledge from the retrieved documents. Finally, a response module will consider relevant context and knowledge while generating a response. 
        * The knowledge selection module utilizes the [Fusion-to-decoder](https://github.com/facebookresearch/FiD) (FID) model, initially designed for open-domain question answering. Kurt’s [EMNLP 2020 paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=k8eeP8EAAAAJ&sortby=pubdate&citation_for_view=k8eeP8EAAAAJ:8k81kl-MbHgC) further applies the FID model and [RAG](https://ai.facebook.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/)to open-domain chit-chat systems and shows impressive improvements. Note that FID fixes the retrieval model during the training process, while RAG jointly trains the model and the retrieval model. 
        * Open-domain question answering models have shown great potential in open-domain chit-chat systems, and the retrieval-augmented models further improve their performances. Is it possible that we can add or design retrieval-augmented models to the task-oriented dialogue systems? As such the system could keep refresh through searching the Internet. 

* [DAIR: Data Augmented Invariant Regularization](https://arxiv.org/abs/2110.11205)
    * Data augmentation techniques on MultiWOZ and SGD datasets. The techniques are also successfully used in the Cairaoke project. 
* [UniGDD: A Unified Generative Framework for Goal-Oriented Document-Grounded Dialogue](https://arxiv.org/pdf/2204.07770.pdf)
* [Commonsense Reasoning for Conversational AI: A Survey of Recent Datasets and Benchmarks](https://openreview.net/forum?id=Dgsu6DVqp5Y)

### Conversational Recommendation:

ARR April 2022

* [RID: A Unified Framework for Conversational Recommender Systems with Pretrained Language Models](https://openreview.net/forum?id=wfFhGDqtIH)

WSDM 2022

* [C2-CRS: Coarse-to-Fine Contrastive Learning for Conversational Recommender System](https://arxiv.org/abs/2201.02732)

CRS Lab

* [A toolkit for conversational recommendation systems](https://github.com/RUCAIBox/CRSLab)

### Question Answering 

[A Memory Efficient Baseline for Open Domain Question Answering](https://arxiv.org/abs/2012.15156)


## 🚀 Week 01 04/22/2022 (will reorganize this part and deleted some to many papers soon)

### Dialogue Related Papers

* ACL 2022 
    * Dialog state tracking:
        * Beyond the Granularity: Multi-Perspective Dialogue Collaborative Selection for Dialogue State Tracking
        * Continual Prompt Tuning for Dialog State Tracking
        * Towards Fair Evaluation of Dialogue State Tracking by Flexible Incorporation of Turn-level Performances
        * ASSIST: Towards Label Noise-Robust Dialogue State Tracking - (Findings of ACL) [Shelby Heinecke](https://salesforce.quip.com/NICAEAJ67wD)
        * Dialogue Summaries as Dialogue States (DS2), Template-Guided Summarization for Few-shot Dialogue State Tracking - (Findings of ACL)
        * N-Shot Learning for Augmenting Task-Oriented Dialogue State Tracking - (Findings of ACL)
    * [DialogVED: A Pre-trained Latent Variable Encoder-Decoder Model for Dialog Response Generation](https://openreview.net/forum?id=WuVA5LBX5zf)
    * Internet-Augmented Dialogue Generation - Kurt
    * Multimodal Dialogue Response Generation
    * ProphetChat: Enhancing Dialogue Generation with Simulation of Future Conversation
    * [SalesBot: Transitioning from Chit-Chat to Task-Oriented Dialogues](https://openreview.net/pdf?id=0Wky3xP0347)
    * UniTranSeR: A Unified Transformer Semantic Representation Framework for Multimodal Task-Oriented Dialog System
    * [UniDU: Towards A Unified Generative Dialogue Understanding Framework](https://arxiv.org/pdf/2204.04637.pdf) 
        * It designs a unified generative framework for dialogue understanding task, including dialogue summary (DS), dialogue completion (DC), slot filling (SF), intent detection (ID) and dialogue state tracking (DST). 
        * The task query can be regarded as the task-specific _prompt_, which includes the task definition and domain-related information. 
        * It shows good _few-shot and zero-shot_ performance. 
        * Deserve to note: 
        * In general, this paper uses a similar architecture to [T0](https://arxiv.org/pdf/2110.08207.pdf), [UnifiedSKG](https://arxiv.org/abs/2201.05966) and [PPTOD](https://arxiv.org/abs/2109.14739). All of them have text-to-text pattern and use multi-task learning. They have shown impressive performance on few-shot and zero-shot learning.
        * The intent name of negative sample is “not defined”, where the input utterances Un are sampled from out-of-domain dialogues. The ratio of negative and positive samples for both DST and ID is set to 2:1. 
        * It is interesting to see whether it will take _a long time to train the model_ and whether _it can only generate pre-defined classes_ rather than random tokens.
        * [Image: image.png]
    * Other papers (low priority):
        * [An Interpretable Neuro-Symbolic Reasoning Framework for Task-Oriented Dialogue Generation](https://arxiv.org/abs/2203.05843)
        * Knowledge Enhanced Reflection Generation for Counseling Dialogues
        * [CICERO: A Dataset for Contextualized Commonsense Inference in Dialogues](https://arxiv.org/pdf/2203.13926.pdf)
    * 
* ACL 2022 - Findings
    * Towards Large-Scale Interpretable Knowledge Graph Reasoning for Dialogue Systems
    * Data Augmentation and Learned Layer Aggregation for Improved Multilingual Language Understanding in Dialogue
    * Multi-Stage Prompting for Knowledgeable Dialogue Generation
* ARR open-review April
    * Highlights:
        * [Towards Building Accurate End-to-End Task-Oriented Dialog Systems with a Simple Cache](https://openreview.net/pdf?id=zbbNFx6cBZJ) ->  [Jason WU](https://salesforce.quip.com/TFDAEAfzfaI) [Huan Wang](https://salesforce.quip.com/KJcAEASCx73)
        * [Commonsense Reasoning for Conversational AI: A Survey of Recent Datasets and Benchmarks ****](https://openreview.net/forum?id=Dgsu6DVqp5Y)
    * [ClidSum: A Benchmark Dataset for Cross-Lingual Dialogue Summarization ****](https://openreview.net/forum?id=6l7l6AkebC)
    * [Navigating Connected Memories with a Task-oriented Dialog System ****](https://openreview.net/forum?id=ktyf0Klfw8)
* ARR open-review March
    * [Learning to Predict Persona Information for Dialogue Personalization without Explicit Persona Description](https://openreview.net/pdf?id=BhzgC_ebxf5)
* ARR open-review Feb
    * [Controllable Multi-attribute Dialog Generation with PALs and Grounding Knowledge](https://openreview.net/forum?id=H6fxZks6qkc)
    * [Simulating Inconsistencies in Task-oriented Dialog](https://openreview.net/forum?id=STMILJiT519)
* ARR open-review Jan
    * [Unsupervised Slot Schema Induction for Task-oriented Dialog](https://openreview.net/forum?id=5moYSLDDnop) → MultiWOZ and SGD
    * [Schema Encoding for Transferable Dialogue State Tracking ****](https://openreview.net/forum?id=RDCgxEa1lgC)
    * [XQA-DST: Multi-Domain and Multi-Lingual Dialogue State Tracking](https://openreview.net/forum?id=r-Ku-qLRgVb)
    * [Learn to Discover Dialog Intents via Self-supervised Context Pretraining](https://openreview.net/pdf?id=AMdwI5DqcMf)
    * [EVI: Multilingual Spoken Dialogue Tasks and Dataset for Knowledge-Based Enrolment, Verification, and Identification](https://openreview.net/forum?id=p5jgs957DXh)
    * Meta AI (Internship Friends)
        * [Knowledge-Grounded Dialogue Generation with a Unified Knowledge Representation](https://openreview.net/forum?id=AY2bywSJyHt)
        * [CheckDST: Measuring Real-World Generalization of Dialogue State Tracking Performance](https://openreview.net/forum?id=I_YteLtAYsM)
        * [KETOD: Knowledge-Enriched Task-Oriented Dialogue ****](https://openreview.net/forum?id=DLKd7j4fThm)
    * [Multi2WOZ: A Robust Multilingual Dataset and Conversational Pretraining for Task-Oriented Dialog](https://openreview.net/forum?id=JhU9onUBeC)
    * [Towards Policy-Guided Conversational Recommendation with Dialogue Acts ****](https://openreview.net/forum?id=wAprE_MK-o-)
    * [Small Changes Make Big Differences: Improving Multi-turn Response Selection in Dialogue Systems via Fine-Grained Contrastive Learning ****](https://openreview.net/forum?id=1U7HCdg9Ed)→ Ubuntu Dialog and Douban corpus
    * [Target-Guided Dialogue Response Generation Using Commonsense and Data Augmentation ****](https://openreview.net/forum?id=XVrgLklgZN)

### Question Answering 

[Towards Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/pdf/2112.09118.pdf)

* It evaluates the models on the [BEIR benchmark](https://github.com/beir-cellar/beir), where the benchmark contains 18 retrieval datasets with a focus on diversity. Most datasets do not contain a training set and the focus of the benchmark is *zero-shot* retrieval. 
* It shows SOTA performances on unsupervised learning and few-shot learning. The unsupervised pre-training alone outperforms BERT with intermediate[MS-MARCO](https://arxiv.org/abs/1611.09268)fine-tuning. 
* Deserve to note:
    * It explores the limits of contrastive learning as a way to train _unsupervised dense retrievers_, and show that it leads to strong retrieval performance. 
    * The ways to build positive pairs and negative pairs are interesting.
        * Building positive pairs from a single document: (1) Inverse Cloze Task: it uses the tokens of the span as the query and the rest of the tokens as the document (or key); (2) Independent cropping: It samples independently two spans from a document to form a positive pair.
        * Building large set of negative pairs: (1) Negatively pairs within a batch based on [SimCLR](https://arxiv.org/abs/2002.05709). (2) Negative pairs across batches where queries are generated from the elements of the current batch and keys are the elements stored in the queue.  The technique is proposed by [MoCO](https://arxiv.org/abs/1911.05722).

[Improving Passage Retrieval with Zero-Shot Question Generation](https://arxiv.org/pdf/2204.07496.pdf)
[LOOPITR: Combining Dual and Cross Encoder Architectures for Image-Text Retrieval](https://arxiv.org/pdf/2203.05465.pdf)
[RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2010.08191.pdf)
[Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks](https://arxiv.org/pdf/2010.08240.pdf)
[Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation](https://arxiv.org/pdf/2010.02666.pdf)
[In-Batch Negatives for Knowledge Distillation with Tightly-Coupled Teachers for Dense Retrieval](https://aclanthology.org/2021.repl4nlp-1.17.pdf)
[Improving Bi-encoder Document Ranking Models with Two Rankers and Multi-teacher Distillation](https://arxiv.org/pdf/2103.06523.pdf)


### Conversational Recommendation Systems

Two tutorials:

* [Tutorials on Conversational Recommendation Systems](https://zuohuif.github.io/RecSys2020ConvRecTutorial/)
*  [Conversational Recommendation: Formulation, Methods, and Evaluation](http://staff.ustc.edu.cn/~hexn/slides/sigir20-tutorial-CRS-slides.pdf) 

WSDM 2022

*  [C2-CRS: Coarse-to-Fine Contrastive Learning for Conversational Recommender System](https://arxiv.org/abs/2201.02732) 

ACL ARR April

* [RID: A Unified Framework for Conversational Recommender Systems with Pretrained Language Models ****](https://openreview.net/forum?id=wfFhGDqtIH)

### Recommendation Systems

### CV & Multi-modal 
 

