***
# Design

## Project Overview

I have chosen the template "Deep Learning on a public dataset" of CM3015 "Machine Learning and Neural Networks". In this project, I am going to build a text summarization model with the CNN dailymail datasets.

Unlike other templates of other courses, the deliverables the machine learning model and its documents, and this project does not have the user interfaces for end users. However, I assume to deliver the machine learning model to an organization that has a plan where the text summarization model is integrated into a system, which has actual end users.

There is also the aspect of a research project where I investigate recent trends and deepen my knowledge of Deep Learning that was not sufficiently covered in the CM3015 course. Day by day and month by month, the field and market of deep learning technology for natural language processing is rapidly growing. The deliverable document is going to be beneficial for developers.

## Domain and Users

As I mentioned above, unlike other project templates, the project template on CM3015 does not have a specific user. However, since it is difficult without specific users in thinking what is beneficial and how beneficial a feature is, I assume that users are developers out of the organization who want to install a machine learning feature and model in to a current running system. Specifically that is as follows.

They are not familiar with deep learning but with server side application of Python, and they want to adopt a text summarization feature into a current running system. My job is solving the problem and providing text summarization models and their knowledgeable and beneficial document to them.

## Justify features based on domain and users

Generally, end users do not mind if a model is highly advanced model or not. Instead, they mind the time and performance of response. 

### Why is the Transformer model used?

In this project, text summarization models are built. That is, the neural network learns sequences as the input to output another sequences as the output. The Transformer model is superior to the Dense model to process sequences data. That is the reason why the Transformer model is used for the text summarization model.

On text classification tasks, well-tuned dense layer models overcome advanced models, such as a Transformer model, an RNN model and so on, under certain conditions.

However, text generation tasks, such as text summarization, could not achieve satisfactory outcomes for a long time. Encoder-decoder models, text embeddings, and Transformer which can handle sequences well have made them possible. That is why the Transformer model is used.

### Why is hyperparameter tuning necessary?

There are a number of hypterparameters in a neural network model, and there does not exist a formula that can derive optimized their values. They depend on each dataset and architecture. The best parameter set must be found with hyperparameter tunings.

Prior to testing advanced models, the hyperparameter tunings should be executed sufficiently. Because even if an advanced model such as the RetNet outperforms a traditional model, it is impossible to fairly compare the advanced model and the traditional model without hyperparameter tunings.

Fairly comparisons derive better models that are beneficial for users. That is the reason why hypter parameter tunings are necessary.

### Why is the RetNet model reseached optionally?

Optionally, the other cutting-edge models, such as the RetNet model, are examined. Generally, end users do not mind if a model is highly advanced model or not. Instead, they mind the time and performance of response. Whereas, it might be able to efficiently use resources, and that might lead to the reduction of the cost for users. This is one of the reason why the research of the RetNet model is made optional.

## Structure of the project

The main deliverable is a model for the text summarization task and its Jupiter notebook document. The model is provided with minimum viable code to run it. Multiple models are possibly provided because the difference in the size of each model affects the machine's requirements where it runs. The notebook's headings will be as follows.

- Abstract
- Introduction
- Background
    - Mathematic explanation
    - Scientific description
    - Technical description
- Experiments/Methodology
- Conclusion
    - Best model
    - Verification of hypotheses
    - Future work
- Citation/Reference

## Key technologies and methods

I am going to introduce libraries and technologies here. From the point of correctness and objectivity, I explain them in my own words as little as possible and I cite public documents, websites, and their Wikipedia page. Instead, I explain how each technology and library is used in my own words for the project.

### Technology

#### Transformer

<img src="img/Screenshot 2024-06-08 at 19.19.07.png">

### Library

#### TensorFlow

This is a machine learning library, which is mainly used in this project.

> An end-to-end open source machine learning platform for everyone. Discover TensorFlow's flexible ecosystem of tools, libraries and community resources. [12]

> TensorFlow is a free and open-source software library for machine learning and artificial intelligence. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks. [13]

In this project, the TensorFlow library is used not only for building text summarization models but also for most other tasks from the beginning to the end.

#### Keras

This is an OSS neural network library. It depends on the library version, it can switch Tensorflow to other machine learning library.

> Keras is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages. Keras also gives the highest priority to crafting great documentation and developer guides. [14]

> Keras is an open-source library that provides a Python interface for artificial neural networks. Keras was first independent software, then integrated into TensorFlow library, and later supporting more. [15]

This library is used throughout the whole project to manipulate the TensorFlow library.

#### KerasNLP

> KerasNLP is a natural language processing library that works natively with TensorFlow, JAX, or PyTorch. Built on Keras 3, these models, layers, metrics, and tokenizers can be trained and serialized in any framework and re-used in another without costly migrations. [16]

In this project, the Transformer model, which consists of the Transformer encoder layer and the Transformer decoder layer, is used to build text summarization models. It is possible to self-implement them by following the paper and books. However, it might unintentionally include bugs. Then, the model and its documents lose the reliability for readers. Therefore, the KerasNLP, which is a part of the widely known library "Keras" and contains the Transformer layer classes, is utilized here to secure the reliability.

Moreover, a lot of hyperparameter tunings are executed in this project. Their classes in KerasNLP is well-architected so that hyperparameters are easily injected to each layer. This is another reason why the KerasNLP is used here.

#### PyTorch

This is another OSS machine learning library. PyTorch is introduced on Wikipedia as follows.

> PyTorch is a machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, originally developed by Meta AI and now part of the Linux Foundation umbrella. It is recognized as one of the two most popular machine learning libraries alongside TensorFlow, offering free and open-source software released under the modified BSD license. Although the Python interface is more polished and the primary focus of development, PyTorch also has a C++ interface. [17]

It is not directly used for the project development. However, some text summarization models that I have introduced has been developed with this PyTorch library.

#### Transformers

This is a software library of Hugging Face, Inc. The official repository of GitHub introduces as follows.

> Transformers provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio. [18]

> Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and can be modified to enable quick research experiments. [18]

> Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration between them. It's straightforward to train your models with one before loading them for inference with the other. [18]

I have investigated the following 2 text summarization models as a part of competitive research.

- facebook/bart-large-cnn [19]
- google/pegasus-cnn_dailymail [20]

They are distributed through huggingface and the Transformers library is necessary to use these models.

## Work Plan (Gantt Chart)

The first 2 to 3 weeks are used to develop the basic Transformer model. Its part of the document is written at the same time. After that, hyperparameter tunings are repeatedly executed with small cycles so that they are efficient. The uncertainty of this phase is the length of training time. The optional implementation for the RetNet model keeps aside the time of 3 weeks. At last, there is a time for the documentation and the buffer.

<img src="img/gantt_chart.png">

## Evaluation Plan

Normal testing for machine learning models and hypothesis verification are executed to evaluate the project. However, the human tester is not used to check performance because it is difficult to organize the tester team for this project free of charge.

### Testing

For the test dataset, the following metrics are used to measure and test model performance. This is executed to numerically prove how better a generated model is.

- loss
- ROUGE-N
- ROUGE-L

### Verification for hypotheses

The following hypotheses are verified.

- Hyperparameter tunings are significantly effective even on a Transformer model.
- Transformer models can generate text more sophisticatedly than dense layer models.
- The encoder-decoder model of the Transformer works better for text summarization tasks than the only-decoder model.
- Pre-trained models improve the text summarization performance.

The following hypothesis is verified as a result of an optional implementation, if possible.

- The RetNet architecture for a small model does not make a difference.

Whereas, it is seemingly difficult to verify the following hypotheses in this project, even though they are interesting topics. The first one is a thought during the prototype implementation. A huge Transformer model is essential to verify the first hypothesis, and it is impossible for the current local environment to run. And no one has solved the second one as far as I have read a number of papers. That is, the task of text summarization is strongly related to human prior knowledge and understanding of the things that are summarized. Thus, I am going to exclude them from the verification of the project temporarily.

- An excessive number of units and layers for the number of dataset entries makes overfitting.
- Prior knowledge and bias cannot be solved on the current architecture.
