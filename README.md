# Innovation Novelty and Firm Value: Deep Learning-based Text Understanding

### ICIS 2023

https://aisel.aisnet.org/icis2023/dab_sc/dab_sc/8/ 

## Abstract:

Innovation is widely acknowledged as a key driver of firm performance, with patents serving as unique indicators of a company’s technological advancements. This study aims to investigate the impact of textual novelty within patents on firm performance, focusing specifically on biotechnology startups listed on the Nasdaq. Utilizing deep learning-based approaches, we construct measures for semantic originality in patent texts. Through panel vector autoregressive (VAR) analysis, our empirical findings demonstrate a positive correlation between textual novelty and abnormal stock returns. Further, impulse response function analysis indicates that the impact of textual novelty peaks approximately one week after patent issuance and gradually diminishes within a month. These insights offer valuable contributions to both the theoretical understanding and practical application of innovation management and strategic planning.

## Authorship:
Chan, Yuk Yee; Hu, Wei; Huang, Jianming; Zhou, Wanyue; Li, Xin

## Keywords:
Innovation, patent, text analysis, textual novelty, stock market, deep learning

## Getting Started

### Dependencies

- Python
  - Python 3.9
  - numpy 1.23.0
  - pandas 1.3.5
  - matplotlib 3.8.2
  - tensorflow 2.10.0
  - torch 2.0.0
  - sentence-transformers 2.2.2
  - tensorflow
  - ipython 8.11.0
  - sklearn

### Setting up the Conda Environment

1. Clone this repository to your local machine.
2. Open a terminal or command prompt and navigate to the project directory.
3. Create a new Conda environment using the following command:

```bash
conda create -n myenv python=3.9
conda activate myenv
conda install -c conda-forge numpy=1.23.0 pandas=1.3.5 matplotlib=3.8.2 tensorflow=2.10.0 torch=1.0.0 sentence-transformers=2.2.2 ipython=8.11.0 scikit-learn
python ./src/main.py
```

## Textual Innovation Novelty Measurement Development:

> The textual novelty detection problem can be framed as follows: Given a new document p and a set of existing documents D={di}, the textual novelty detection is to define a function TN(p, D) that tells how novel p is given the existence of D. In this project, we develop three methods for textual novelty.

<img src='./fig/Novelty Measurement.png' width='800'>

### 1. TFIDF-based Maximum Similarity Method (TFIDF-based)

<img src='./fig/TDITF.png' width='600'>

We define similarity as the cosine similarity built upon the TF-IDF vector representation of documents. After transforming each document into a vector of TF-IDF values, the cosine similarity of any pair of vectors is obtained by taking their dot product and dividing it by the product of their norm.  

### 2. Bert-based Maximum Similarity Method (BERT-based)

<img src='./fig/BERT.png' width='600'>

We use SBERT, a modification of BERT, to generate meaningful sentence embeddings. These embeddings are compared using cosine similarity. We chose the "sentence-transformers/all-MiniLM-L6-v2" model, which maps sentences and paragraphs to a fixed 384-dimensional vector space. This vector space can be used for tasks like semantic search. Then, we calculate similarity using cosine similarity. In addition, we normalize the novelty scores without changing their magnitude for better representation of the data.

> For fine-tuning the BERT-based word vector model, we split the patent data into sentence-level training data. We set the number of warm-up steps for the SBERT model to 500, the training batch sample size to 32, the epochs step size to 10, and leave the rest of the parameters as default settings. After training the model, we obtain a word vector representation model.

### 3. Variational AutoEncoding (VAE)

<img src='./fig/VAE.png' width='600'>

Autoencoders use the same data for input and output layers to learn dataset representations, often for dimensionality reduction and eliminating unwanted signals. Normal inputs can pass through layers with minimal loss, but novel inputs deviating from hidden patterns will experience greater data loss. To obtain novelty scores, we utilize the Variational Autoencoder (VAE). VAE is an autoencoder with a regularized latent distribution. During training, it samples from a normal distribution to ensure a well-characterized latent space, leading to improved results.

#### Architecture of Variational Autoencoder model

<img src='./fig/VAE_model2.png' width='600'>

## Evaluation Framework

> We set up a baseline document set with normal (non-novel) documents and two comparison groups, one with only normal documents and one with both normal and novel documents. The former is simulating the situation of the occurrence of normal documents, and the latter is simulating the situation of the occurrence of novel documents; see Figure as follows:

<img src='./fig/Evaluation_framework.png' width='600'>

## Performance Comparison

<img src='./fig/TFIDF_dist.png' width='600'>

When looking at the different methods used, the TFIDF-based method showed that most novel classes seem to have novelty scores between 0.8 and 1, whereas normal classes seem to have an irregular distribution. 

<img src='./fig/BERT_dist.png' width='600'>

<img src='./fig/VAE_dist.png' width='600'>


BERT-based and VAE methods showed normal distribution for both normality and novelty classes. However, the VAE method had a more centralized normality class and the BERT-based method had a more centralized novelty class. This suggests that the BERT-based method may be more sensitive in identifying unique features in novel documents, while the VAE method may be better at distinguishing between normal and novel classes.

### Measure Performance Comparison

<img src='./fig/Performance_comparison.png' width='600'>

The Bert-based Maximum Similarity method has the highest correlation coefficient of 0.808 among the three methods, indicating a strong positive correlation between the predicted novelty scores and the novelty/normal classes. 

The KS test of the Bert-based Maximum Similarity method has a value of 0.865, the highest among the three methods. This indicates that the distribution of novelty scores for the novelty/normal classes produced by the Bert-based method is significantly different compared to the others. 


> Pearson correlation coefficient: To determine the difference between normal and novel documents, we calculate the correlation between the novelty score and the novelty or normal classes.

> Kolmogorov-Smirnov test: It can be used to compare the distributions of two comparison groups to determine if they are significantly different from each other.

> Jaccard coefficient: It measures the similarity or overlap level between two comparison groups.

> Jaccard coefficient (continuous version): The Jaccard coefficient in the continuous version.

> Dice coefficient: It measures the similarity or overlap level between two comparison groups.

### Panel VAR Model
This study uses a panel VAR model to analyze the relationship between a firm's textual patent novelty and its performance. The findings suggest that innovation novelty has a significant impact on firm value, with investors quickly responding to pioneering patents in the biotechnology sector within a week.

<img src='./fig/IRF.png' width='800'>

To quantify the effect of the change in dependent variables lagged more than one period, impulse response functions (IRFs) are often used to visually interpret the coefficient estimates generated for panel VAR models by simulating the fitted panel VAR model through a Monte Carlo simulation with 1,000 runs. As shown in Figure, the IRF results reveal that the effect of the textual patent novelty score on firm value lasts for about one week and gradually decreases to zero as the effect eventually dies out.

### Conclusion

This study introduces deep learning-based approaches, utilizing both VAE and BERT algorithms, to measure textual patent novelty and examine its impact on firm value. Employing the panel VAR model, we find that textual patent novelty positively influences firms’ abnormal returns. The impulse response functions (IRFs) results reveal that the impact of textual patent novelty on firm performance peaks within one week and then diminishes within one month. Our findings underscore the significance of innovation and enrich our understanding of patent text by providing a state-of-the-art measure for innovation.

## Overview of our Research
See documentation [here](./PPT.pdf)

## Textual Measure Report
See documentation [here](./Report.pdf)

## Final Research paper
See documentation [here](./icis23a-sub1249-cam-i9.pdf)
