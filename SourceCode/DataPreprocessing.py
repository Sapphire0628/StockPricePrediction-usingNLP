import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from IPython.display import Image
import re
import statistics
import scipy
from statistics import mean 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import FastICA 
from sklearn.metrics import classification_report
from sklearn import metrics

# Function Definition

# 删除电子邮件
def remove_email(text):
    url_pattern = re.compile(r"[A-Za-z0-9+-\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+")
    return url_pattern.sub(r'', text)

# 删除标点符号
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    for character in string.punctuation:
        text = text.replace(character, ' ')
    return text

# 删除停用词
def remove_stopwords(text):
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def divideClass(_knownClass,_noveltyClass):
    from sklearn.datasets import fetch_20newsgroups
    # Classify the Known class and the Novelty class 对已知类和新奇类进行分类

    # 已知类 : alt.atheism , comp.graphics , comp.os.ms-windows.misc
    #         (comp.sys.ibm.pc.hardware , comp.sys.mac.hardware , comp.windows.x)
    # 新奇类 : misc.forsale
    newsgroups_known = fetch_20newsgroups(subset='train', categories=_knownClass)
    newsgroups_novelty = fetch_20newsgroups(subset='train', categories=_noveltyClass)
    known_dict = {"text": newsgroups_known.data, "class": newsgroups_known.target}
    known_data = pd.DataFrame(known_dict)
    novelty_dict = {"text": newsgroups_novelty.data, "class":len(_knownClass)}
    novelty_data = pd.DataFrame(novelty_dict)
    #print(known_data["text"][0])
    # Data preprocessing (remove_email, remove_punctuation,remove_stopwords)
    known_data["text"] = known_data["text"].apply(lambda text: remove_email(text)).str.lower().apply(lambda text: remove_punctuation(text)).apply(lambda text: remove_stopwords(text))
    novelty_data["text"] = novelty_data["text"].apply(lambda text: remove_email(text)).str.lower().apply(lambda text: remove_punctuation(text)).apply(lambda text: remove_stopwords(text))
    #print(known_data["text"][0])
    print("Length of Known dataset : ", len(known_data))
    print(f"Known dataset incluced ",known_data["class"].value_counts().shape[0]," class.")
    print()


    print("Length of Novelty dataset : ", len(novelty_data))
    print(f"Novelty dataset incluced ",novelty_data["class"].value_counts().shape[0]," class.")
    return known_data,novelty_data

def mergeDf(Novelty, Normality):
    data = pd.concat([Novelty, Normality], ignore_index=True)
    X = data[["Max","Std"]]
    Y = data["Target"]
    """
    Novelty_plot = plt.scatter(Novelty["Max"], Novelty["Std"],color='r')
    Normality_plot = plt.scatter(Normality["Max"], Normality["Std"],color='g')

    plt.legend((Novelty_plot, Normality_plot),
               ('Novelty', 'Normality'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    
    plt.title("Realistic graph")
    plt.xlabel('Max')
    plt.ylabel('SD')
    
    plt.show()
    """
    return X,Y
