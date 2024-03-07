def openFile():
    import json
    f = open('./PatentData/all_file.json')
    data = json.load(f)
    f.close()
    return data



def processingDF(data):
    import pandas as pd
    textdata = pd.DataFrame(columns=["text","label"])
    for i in data:
        if i["abstract"] == None:
            print(i["file"].split("-")[0]+"'s abstract does not exist.")
        else: 
            for j in range(len(i["abstract"].split(". "))):
                textdata = textdata.append({"text":i["abstract"].split(". ")[j],"label":0},ignore_index=True)
    return textdata

data = openFile()
textdata =processingDF(data)
#%%
from sentence_transformers import InputExample
train_data = []

for i in range(len(textdata)):
    train_data.append(InputExample(texts=[textdata["text"][i],textdata["text"][i]]))
    
#%%
    
# 定义微调的dataset、dataloader、loss
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses, models
from torch.utils.data import DataLoader
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

train_dataset = SentencesDataset(train_data, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
# 无监督fintune的loss类型，不需要标签
train_loss = losses.MultipleNegativesRankingLoss(model)
# 微调训练
model.fit(train_objectives=[(train_dataloader, train_loss)],show_progress_bar=True, epochs=10, warmup_steps=500, output_path='./FYP_BERT_Model')
