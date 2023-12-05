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

![image]("./Figure/BERT.png") 

We define similarity as the cosine similarity built upon the TF-IDF vector representation of documents. After transforming each document into a vector of TF-IDF values, the cosine similarity of any pair of vectors is obtained by taking their dot product and dividing it by the product of their norm.  

### 2. Bert-based Maximum Similarity Method

### 3. Variational Autoencoding

## Evaluation Framework
See documentation [here](./PPT.pdf)

## Textual Measure Report
See documentation [here](./Report.pdf)

## Final Research paper
See documentation [here](./icis23a-sub1249-cam-i9.pdf)
