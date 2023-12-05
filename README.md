# Innovation Novelty and Firm Value: Deep Learning-based Text Understanding

## Abstract:

Innovation is widely acknowledged as a key driver of firm performance, with patents serving as unique indicators of a company’s technological advancements. This study aims to investigate the impact of textual novelty within patents on firm performance, focusing specifically on biotechnology startups listed on the Nasdaq. Utilizing deep learning-based approaches, we construct measures for semantic originality in patent texts. Through panel vector autoregressive (VAR) analysis, our empirical findings demonstrate a positive correlation between textual novelty and abnormal stock returns. Further, impulse response function analysis indicates that the impact of textual novelty peaks approximately one week after patent issuance and gradually diminishes within a month. These insights offer valuable contributions to both the theoretical understanding and practical application of innovation management and strategic planning.

## Authorship:
Chan, Yuk Yee; Hu, Wei; Huang, Jianming; Zhou, Wanyue; Li, Xin

## Keywords:
Innovation, patent, text analysis, textual novelty, stock market, deep learning

## Textual Novelty Measure Development

The textual novelty detection problem can be framed as follows: Given a new document p and a set of existing documents D={di}, the textual novelty detection is to define a function TN(p, D) that tells how novel p is given the existence of D. In this project, we develop three methods for textual novelty.
## Overview of our Research

### 1. TFIDF-based Maximum Similarity Method

![image](./Figure/TDIDF.png) 

We define similarity as the cosine similarity built upon the TF-IDF vector representation of documents. After transforming each document into a vector of TF-IDF values, the cosine similarity of any pair of vectors is obtained by taking their dot product and dividing it by the product of their norm.  

### 2. Bert-based Maximum Similarity Method

![image](./Figure/BERT.png) 

We use SBERT, a modification of BERT, to generate meaningful sentence embeddings. These embeddings are compared using cosine similarity. We chose the "sentence-transformers/all-MiniLM-L6-v2" model, which maps sentences and paragraphs to a fixed 384-dimensional vector space. This vector space can be used for tasks like semantic search. Then, we calculate similarity using cosine similarity. In addition, we normalize the novelty scores without changing their magnitude for better representation of the data.

For fine-tuning the BERT-based word vector model, we split the patent data into sentence-level training data. We set the number of warm-up steps for the SBERT model to 500, the training batch sample size to 32, the epochs step size to 10, and leave the rest of the parameters as default settings. After training the model, we obtain a word vector representation model.

### 3. Variational Autoencoding

![image](./Figure/VAE.png) 

Autoencoders use the same data for input and output layers to learn dataset representations, often for dimensionality reduction and eliminating unwanted signals. Normal inputs can pass through layers with minimal loss, but novel inputs deviating from hidden patterns will experience greater data loss.

To obtain novelty scores, we utilize the Variational Autoencoder (VAE). VAE is an autoencoder with a regularized latent distribution. During training, it samples from a normal distribution to ensure a well-characterized latent space, leading to improved results.

### Architecture of Variational Autoencoder model


## Evaluation Framework
See documentation [here](./PPT.pdf)

## Textual Measure Report
See documentation [here](./Report.pdf)

## Final Research paper
See documentation [here](./icis23a-sub1249-cam-i9.pdf)
