***
# Introduction

In this project, I'm going to use CNN daily mail datasets to build a text summarization model. Text summarization task means generating a short organized summary from a long article, such as news, through extracting and abstracting.

This will be a research project in which I investigate recent trends and deepen my knowledge of Deep Learning, which goes beyond the contents of CM3015. A text summarization model is eventually built.

In addition, this project supposes an organization or a group is the user, such as a (web) system development company with little machine-learning knowledge. Therefore, the deliverable report must be scientific and convincing to explain why each parameter is adopted and be knowledgeable for its developers. The deliverable code and model must run easily on common devices, such as a web server. Mobile devices, on the other hand, might not be able to run the large model because they have a smaller amount of memory.

## What is interesting and motivative (Rapid Evolution) 

In recent years, various types of text representations and neural network models have been developed, especially since the Transformer model was invented.

### Text representation

In addition to the One-hot-encoding introduced in CM3015, I am going to use other techniques, such as word2vec and GloVe, that can appropriately handle the meanings of each word.

The neural network is a vector transformation. It is worth investigating how well each representation of embedding layers can capture the characteristics of text data.

### Transformer

This is one of the most common and game-changing models for sequential data, such as text. In addition, it is also used in text-to-image generative AI, LLM, and so on. There are a lot of 
derived models based on the Transformer architecture and one of them is BERT, which is superior to natural language processing. [10] Another model based on the Transformer, named Vision Transformer, has been used for image recognition tasks to overcome CNN's weaknesses recently. [9] Moreover, there is a more efficient model based on the Transformer model, which is called the Retentive Network. Because all these architectures are based on the Transformer model, deepening its knowledge leads to the understanding of whole machine learning. And, even in the future, a number of new architectures based on the Transformer models will be released. That makes me motivated more.

## What is interesting and motivative (from the aspect of efficiency and ecology)

Measuring the performances of various models is one of the exciting and essential points.

Advanced models do not always achieve the best performance. For example, when the Recurrent Neural Network model, which is one of the advanced models, is used for a text classification task, even though it does not always work better than a dense model, it mostly consumes greater computing resources. [11] That is, it completely wastes resources. I am going to analyze the strengths and weaknesses of each model in terms of practical uses.

As recent investigations indicate, building LLM models indirectly emits a significant amount of CO2. That is, the investigation of efficiency that leads to the ecology is socially meaningful, even in the environmental aspect.

However, since there are not enough computing resources to build an LLM model, this project does not target building an LLM itself.
