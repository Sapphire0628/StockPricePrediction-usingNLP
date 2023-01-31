def BERT_Max_Similarity(known_data,novelty_data):
    from transformers import AutoTokenizer, AutoModel
    from sklearn.model_selection import train_test_split
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    bertModel = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    known_train, known_test, y_train, y_test = train_test_split(known_data["text"], known_data["class"], test_size=(novelty_data.shape[0]/(novelty_data.shape[0]+known_data.shape[0])), random_state=42)
    novelty = bertModel.encode(list(novelty_data['text'])+list(known_data['text']))
    normal = bertModel.encode(list(known_test) + list(known_train))


    socres_novelty = cosine_similarity(novelty[:novelty_data.shape[0]],novelty[novelty_data.shape[0]:])
    socres_normality = cosine_similarity(normal[:known_test.shape[0]],normal[known_test.shape[0]:])
    
    return socres_novelty,socres_normality