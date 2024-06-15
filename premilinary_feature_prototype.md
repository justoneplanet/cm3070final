***
# Feature Prototype

First, there are three types of Transformer models: Transformer encoder-only model, Transformer decoder-only model, and Transformer encoder-decoder model. Here are three types of different machine-learning tasks with the Transformer for each type.

1. The Transformer encoder-only model for the text classification task
2. The Transformer decoder-only model for the text generation task
3. The Transformer encoder-decoder model for the text summarization task

Generally, the encoder-decoder model of the Transformer is used for text summarization tasks. However, the decoder-only model might be able to be used. In any case, there is a necessity to verify whether each model and each API of libraries can actually work. This time, the KerasNLP library is used to add the encoder and decoder layers of the Transformer. The reasons why the KerasNLP library is utilized are as follows.

- Stability
- Concise of parameter experiments

The book "Deep Learning with Python, Second Edition" [7] introduces the simplified Transformer implementation that actually works in the local environment. However, it is not certain how widely it is used, and it might not pass any testing. In addition, even though it is good for us to understand how the Transformer works internally because it is simplified, it is difficult for us to experiment with various parameters.

## Transformer text classification task

Firstly, a Transformer classification model is built so that the local environment determines if it can run or not.

## Transformer text generation task

Now, the following implementation, which is partially based on the book "Deep Learning with Python" [6], inspects how the Transformer decoder-only model works, and investigates how long a text generation task takes so that the time for the text summarization task is estimated. The points of difference from the above classification task are mainly as follows.

- The number of units at the output layer.
- The number of layers and units at the hidden layers.
- The number of epochs

In the generative task, such as text summarization, these numbers generally get larger than the text classification model. As a result, it takes a significant amount of time for training. In this case, it took over 8 hours. The number of parameters will be increased at least 15% and the training time will be longer 15%. That is one of the reasons why I have changed my plan, where the RetNet model is an optional topic. Even though the Retentive Network model will make the training 8.4 times faster in the aspect of the throughput, which is shown in the paper, the same Transformer model training must be executed for the comparison. [3] The experiments will still take a lot of time, and that is not realistic.

## Transformer text summarization task

Finally, to understand the outline, the minimum viable implementation of text summarization task is below. Because the dataset contains only 2 entries, this model overfits 100%. This time, the encoder and decoder model of the Transformer is adopted. However, this is not a requirement of the text summarization model. It is not written that the decoder-only model of the Transformer can neither build the text summarization model nor achieve better performance. This will be an experiment in this report.

Moreover, there are various types of text representation/vectorization as follows, including pre-trained models.

- Static embeddings
    - word2vec [22]
    - fastText (more advanced than word2vec) [23]
    - GloVe [24]
- Dynamic embeddings
    - BERT

This is an important experiment. Because, if there exists an ultimate representation to express words and text sequence, the neural network holds the ability to fit the hyper-dimensions. That is, finding better vector representation is essential to improve the performance.

Furthermore, there are many hyperparameters. Some of them are shown at the top of the following code.

### Embedding layer

The default Embedding layer of the Keras library is used in this project. [21] This layer vectorizes words in a sequence, and similar words are vectorized closely. An accurate description is on the official page, as follows.

> Word embeddings give us a way to use an efficient, dense representation in which similar words have a similar encoding. Importantly, you do not have to specify this encoding by hand. An embedding is a dense vector of floating point values (the length of the vector is a parameter you specify). Instead of specifying the values for the embedding manually, they are trainable parameters (weights learned by the model during training, in the same way a model learns weights for a dense layer). It is common to see word embeddings that are 8-dimensional (for small datasets), up to 1024-dimensions when working with large datasets. A higher dimensional embedding can capture fine-grained relationships between words, but takes more data to learn. [25]

This layer has the following tunable hyperparameters.

- The number of vocabulary
- The number of embedding dimension
- Whether the value 0 is masked or not as padding

### PositionEmbedding layer

The following sample code uses the PositionEmbedding layer of the KerasNLP library. [26] The KerasNLP library also has some positional embedding layers, such as the SinePositionEncoding layer that was originally used in the thesis. [27] [1] Whereas because the tunable hyperparameters are few, this project examines various types of positional embeddings here.

- The number of sequence length

### TransformerEncoder layer

The default TransformerEncoder layer of the Keras library is used in this project. [28] This layer encodes text to the meaningful representation internally.

- The number of layers
- The number of hidden units
- The number of heads
- The dropout rate
- The epsilon value for normalization

### TransformerDecoder layer

The default TransformerDecoder layer of the Keras library is used in this project. [29] This layer decodes the internally meaningful representation to the text.

- The number of layers
- The number of hidden units
- The number of heads
- The dropout rate
- The epsilon value for normalization

### Model

- The number of epochs
- The type of optimizer
- The learning rate of the optimizer

Unlike the classification task, the first word is predicted as the decoder output, and it is used for the second word prediction as the decoder input. By repeating this until the decoder outputs the special end symbol, the complete summarized sentence is generated.

