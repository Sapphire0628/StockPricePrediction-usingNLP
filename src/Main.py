import DataPreprocessing
from TFIDF_Sim import TFIDF_Max_Similarity
from BERT_Sim import BERT_Max_Similarity

import Metric
#%%
# Classify the Known class and the Novelty class 对已知类和新奇类进行分类

known_data, novelty_data = DataPreprocessing.divideClass(['alt.atheism',
                                         'comp.graphics',
                                         'comp.os.ms-windows.misc',
                                         ]
                                        ,['sci.med'])

#%%

# TDIDF Methods
TFIDF_socres_novelty,TFIDF_socres_normality = TFIDF_Max_Similarity(known_data,novelty_data)
TFIDF_Novelty, TFIDF_Normality= Metric.NoveltyScore(TFIDF_socres_novelty,TFIDF_socres_normality,"TFIDF")
TFIDF_X,TFIDF_y = DataPreprocessing.mergeDf(TFIDF_Novelty, TFIDF_Normality)

Metric.Correlation(TFIDF_X["Max"], TFIDF_y)
Metric.Overlap(TFIDF_Normality["Max"].round(2), TFIDF_Novelty["Max"].round(2))


#%%
# BERT Methods
BERT_socres_novelty,BERT_socres_normality = BERT_Max_Similarity(known_data,novelty_data)
BERT_Novelty, BERT_Normality= Metric.NoveltyScore(BERT_socres_novelty,BERT_socres_normality,"BERT")
Bert_X,Bert_y = DataPreprocessing.mergeDf(BERT_Novelty, BERT_Normality)

Metric.Correlation(Bert_X["Max"], Bert_y)
Metric.Overlap(BERT_Normality["Max"].round(2), BERT_Novelty["Max"].round(2))

#%%
# Autoencoding 
from AUTO import AutoEncoder
AUTO_socres_novelty,AUTO_socres_normality = AutoEncoder(known_data, novelty_data)
AUTO_Novelty, AUTO_Normality= Metric.NoveltyScore(AUTO_socres_novelty,AUTO_socres_normality,"AUTO")
AUTO_X,AUTO_y = DataPreprocessing.mergeDf(AUTO_Novelty, AUTO_Normality)
Metric.Correlation(AUTO_X["Max"], AUTO_y)
Metric.Overlap(AUTO_Normality["Max"].round(2), AUTO_Novelty["Max"].round(2))


