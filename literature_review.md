***
# Literature Review

I aim to explore the current understanding of machine learning model architectures and their performances, focusing on the key technology "Transformer". Moreover, in this project, I investigate the effectiveness of the Transformer and finally build a text summarization model.

## Transformer

Vaswani et al. (2017) proposed a new simple architecture, named Transformer, based on attention mechanisms without recurrence and convolutions. [1]

The architecture of the Transformer, which consists of both encoders and decoders, is introduced with a simple figure.

The paper indicates the algorithms and architecture of the Transformer with formulas, comparing the former architectures such as the recurrent neural network model and the gate recurrent unit model. How to reduce the amount of calculation for longer input sequences, which is one of the defects, is also introduced kindly. However, because of the nature of the paper, the concrete implementation is not shown on it. Instead, it provides a link to a GitHub repository, "tensor2tensor," [2] where some implementations with Python code are introduced.

Though these codes might still be beneficial, They have not been maintained for years. It is pretty uncertain whether it can run on the current Python environment.

Because the Transformer was initially developed for the machine translation task, various tasks except translation are not mentioned in this paper, written in 2017. Thereby, the effectiveness of the Transformer is not investigated for a wide range of NLP tasks here.

As mentioned at the beginning, hyperparameter tunings are obviously required to investigate effectiveness. The section "6.2 Model Variations" describes the parameters such as the number of heads, the parameters of the optimizer, and so on that exist in the Transformer model, and the same indicators will be used to compare the performance in this project. And the experiment environment, which the paper disclosed, is useful to the experiments in this project.

I have obtained the following beneficial knowledge, even though it is not directly related to my project.

- The number of heads is neither too many nor too few
- Interpretable models

There are not following items that are essential for this project in the paper.

- Sufficient benchmarks for various NLP tasks/applications
- Minimum viable and runnable code in the modern environment

## RetNet

Yutao et al. (2023) proposed the Retentive Network (RetNet), which is a strong successor to Transformer for large language models.[3]

One of the drawbacks of the Transformer, which is introduced in this paper, is the slow inference and inefficiency due to multi-head attention.
The performance comparison between the RetNet and the Transformer has been executed for large language models and represents the improvement. They are executed and compared for large language models and that difference might not be remarkable on smaller models such as text summarization tasks.

The section "3 Experiments" mentions the following:

> Perplexity decreases along with scaling up the model size. We empirically observe that RetNet tends to outperform Transformer when the model size is larger than 2B. [3]

When the model size is larger than 2.7B, with more than 2.7 billion parameters, the advantages of the RetNet model might finally be detected.

The section "3.3 Training Cost" shows the specific environment where the evaluation of this paper was executed as follows.

> We evaluate the results with eight Nvidia A100-80GB GPUs, because FlashAttention is highly optimized for A100. Tensor parallelism is enabled for 6.7B and 13B models.

Building a text summarization model in this project is mainly executed on the local macOS machine, which has an M2 CPU, which has a significantly different architecture from Nvidia GPUs.

> Moreover, without relying on specific kernels, it is easy to train RetNet on other platforms efficiently. For example, we train the RetNet models on an AMD MI200 cluster with decent throughput. It is notable that RetNet has the potential to further reduce cost via advanced implementation, such as kernel fusion.

Even though the above quote mentioned the performance on the other device, the environment is highly parallelized, and it is uncertain if this project can derive beneficial knowledge through experimental comparisons.

Though this does not matter directly to this project, as the sections "2.3 Overall Architecture of Retention Networks" and "3.4 Inference Cost" mention, the inference cost of the RetNet model is remarkably $O(1)$. [3] In the case where the quick response for longer sequences/tokens is required, the RetNet model can be one of the realistic choices.

## Neural Text Summarization: A Critical Evaluation

Wojciech et al. (2019) introduced and evaluated the current shortcomings of text summarization.[4] The methodology of the evaluation should be notable in this paper. The previously introduced 2 papers are not for text summarization tasks, and the BLEU metric is used, which is for the evaluation of translation task performance. This is not for summarization tasks. Though loss values probably evaluate the performances, they cannot clearly consider sentences' structures for evaluations. Therefore, following this paper, the ROUGE metric is used in this project. Fortunately, KerasNLP has the following 2 metrics as standard.

- [ROUGE-N](https://keras.io/api/keras_nlp/metrics/rouge_n/)
- [ROUGE-L](https://keras.io/api/keras_nlp/metrics/rouge_l/)

Note that the ROUGE-N metric uses n-gram to refer to the overlap, and the ROUGE-L metric uses the longest common subsequence that can detect the sentence level structure. [8]

Though the following doesn't matter directly, some shortcomings of the current summarization model are introduced here.

- Even though a summarization task depends on the reader's expectations and prior knowledge, the current models are not provided as additional information.
- Even though the ROUGE metric is widely used, it does not factually and consistently examine summarized text, but just lexically.
- The bias and diversity of the dataset.

## Text Summarization with Pretrained Encoders

Yang et al. (2019) indicated the effectiveness of the BERT, which is an abbreviation of Bidirectional Encoder Representations from Transformers, for text summarization, which is one of the NLP tasks. [5]

In this paper, the BERT model, where a pretrained model and a scratch model are combined, is compared with the other typical models for text summarization tasks, and it is concluded that the BERT model outperformed others. However, it seems slightly unclear whether the pretrained model actually works better, because there is not a comparison of the same architecture model between pretrained and scratch.

As the section "Introduction" mentions, the BERT model, which is one of the Transformer models and the abbreviation of "Bidirectional Encoder Representations from Transformers", has affected the word and sentence representation. That is, this model possibly becomes a research target in this project.

> When fine-tuning for a specific task, unlike ELMo whose parameters are usually fixed, parameters in BERT are jointly fine-tuned with additional task- specific parameters. [5]

Whereas, as the above is written, since the pretrained BERT model goes through the fine-tuning phase, it might take longer time for training than fixed parameters' models such as word2vec and GloVe.

The table of the section "3.3 Abstractive Summarization" holds the various type of datasets.
Though this might not matter to the paper, apart from CNN DailyMail, NYT, and XSum are used as the datasets. In this project, the CNN dailymail is going to be used as the dataset. However, they are kept as sub plans, preparing for unexpected situations.

> Each token $w_i$ is assigned three kinds of embeddings: token embeddings indicate the meaning of each token, segmentation embeddings are used to discriminate between two sentences (e.g., during a sentence-pair classification task) and position embeddings indicate the position of each token within the text sequence. [5]

If the BERT model is adopted, the differences of embeddings must be paid attention.

## facebook/bart-large-cnn

This is a text summarization model developed by facebook that is available on Hugging Face. [30]

Particularly noteworthy is the model size and its performance. The model size, the number of parameters, is 406M, and the results of ROUGE-1, ROUGE-2, and ROUGE-L are 42.949, 20.815, and 30.619 respectively.

In addition, pre-trained BART is used in this text summarization model.

> BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder. BART is pre-trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text. [30]

> BART is particularly effective when fine-tuned for text generation (e.g. summarization, translation) but also works well for comprehension tasks (e.g. text classification, question answering). This particular checkpoint has been fine-tuned on CNN Daily Mail, a large collection of text-summary pairs. [30]

## google/pegasus-cnn_dailymail

This is another text summarization model developed by Google that is available on Hugging Face. [31]

This model has been trained not only with the CNN dailymail dataset but also with many others. It has 568M parameters and consists of 12 Transformer encoder layers and 12 Transformer decoder layers. The sinusoidal method is used for the positional embedding. The number of vocabulary is 50265.

In addition, the paper "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization" shows that the results of ROUGE-1, ROUGE-2, and ROUGE-L are 44.16, 21.56, and 41.30 respectively. [32]

## Conclusion

In conclusion, the superiority and effectiveness of the Transformer and its self-attention have been shown, compared with the other architectures such as the recurrent neural network and convolutional neural network. Whereas performances, tunings, other NLP tasks, and architectures, which determine if encode or/and decode is used, have not been shown. And that is what I aim to do.

I have slightly changed my mind while reading papers. Specifically, instead of regarding the RetNet model as the main topic, hyperparameter tunings of the Transformer model are going to be mainly researched. The RetNet model will be researched optionally. The reason is as follows.

- The RetNet paper shows that it is effective only for large language models.
- The experiments on the ResNet paper were executed on the highly parallelized environment, which is different from the local machine that is used in this project.
- The training to build a text summarization model takes a considerable amount of time.

That is, because there is a possibility where a beneficial report cannot be provided if most of resources are invested to experiments with little chance of seeing differences, I am going to make the experiment about the RetNet model optional. Instead, clearly beneficial hyperparameter tunings are executed in the first half to priorly ensure the benefits for users.

Moreover, the comparison between the pre-trained models and non-pre-trained models will also be one of the interesting topics.

Finally, it might be very difficult to self-build a text summarization model and mark a better score of ROUGE metrics on the current local machine because the models of Google and Facebook are trained with a number of layers and parameters using high computational GPUs. However, it might be possible to build a smaller model that is beneficial for a specific environment. Since KerasNLP provides an easy way to use BART, the performance might be able to get close to the Google or Facebook results if it is used. [33]
