import json
import pandas as pd 
import numpy as np
from datetime import datetime
#%%
# Opening JSON file
def openFile():
    f = open('./PatentData/all_file.json')
    data = json.load(f)
    f.close()
    return data



def processingData(data):
    df = pd.DataFrame(columns=["patentID","text","bert_encode","TFIDFnoveltyScore","BERTnoveltyScore"])
    
    for i in data:
        if i["abstract"] == None:
           continue
        else: 
            patentID = i["file"].split("-")[0]
            date = i["file"].split("-")[1].split(".")[0]
            text = i["abstract"]
            df = df.append({"date":datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d'),"patentID":patentID,"text":text}, ignore_index=True)
    return df

data = openFile()
patentData = processingData(data)

#%%

# Data preprocessing (remove_email, remove_punctuation,remove_stopwords)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util



tfidf = TfidfVectorizer(binary=True)
bertModel = SentenceTransformer("./FYP_BERT_Model")
for n, doc in enumerate(patentData["text"].iloc):    
    patentData["bert_encode"][n] = bertModel.encode(doc)
#%%
for n , patent in enumerate(patentData.iloc):
    pre_patent = patentData[patentData['date'] < patent["date"]]
    if pre_patent.empty:
        continue
    else:
        
        # TFIDF
        pre_tfidf = tfidf.fit_transform(pre_patent["text"])
        cur_tfidf = tfidf.transform(patent[["text"]]) 
        noveltyScore = 1 - cosine_similarity(cur_tfidf,pre_tfidf).max()
        if noveltyScore < 0 :
            noveltyScore = 0
        patentData.loc[n,"TFIDFnoveltyScore"] = noveltyScore
        
        
        pre_bert = []
        for i in pre_patent["bert_encode"].values:
            pre_bert.append(i)
        pre_bert = np.array(pre_bert)
        cur_bert = patent["bert_encode"].reshape((1,384))
        BERTnoveltyScore = 1 - cosine_similarity(cur_bert,pre_bert).max()
        if BERTnoveltyScore < 0 :
            BERTnoveltyScore = 0
        patentData.loc[n,"BERTnoveltyScore"] = BERTnoveltyScore
        
        
#%%
        
        
        

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
# Scaling BERT novelty score columns
patentData['BERTnoveltyScore'] = mms.fit_transform(patentData[['BERTnoveltyScore']]).astype('object')
patentData = patentData.dropna()
patentData = patentData.drop(columns=["text","date","bert_encode"])
patentData.to_json('PatentNovelty.json', orient='records', lines=True)
