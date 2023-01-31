def TFIDF_Max_Similarity(known_data,novelty_data):
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics.pairwise import cosine_similarity
    # 比较并收集 1.已知数据集 和 2.新奇数据集 之间的差异分数

    # 1. 使用已知数据集 (Seen) 拟合 Tfidf 模型, 然后用 新颖数据集 (novelty) 按照训练数据同样的模型进行转换，得到特征向量
    tfidf_1 = TfidfVectorizer(binary=True)
    Seen = tfidf_1.fit_transform(known_data["text"]) 
    novelty = tfidf_1.transform(novelty_data["text"]) 

    #print("                      (observations, vectors)")
    #print("Seen Feature vector :        ", Seen.shape)
    #print("Novelty Feature vector :       ", novelty.shape,"\n")

    # 2. 使用已知数据集 (known_train) 拟合 Tfidf 模型, 然后用 相似数据集 (known_test) 按照训练数据同样的模型进行转换，得到特征向量
    tfidf_2 = TfidfVectorizer(binary=True)
    known_train, known_test, y_train, y_test = train_test_split(known_data["text"], known_data["class"], test_size=(novelty.shape[0]/(novelty.shape[0]+Seen.shape[0])), random_state=42)
    #known_train, known_test, y_train, y_test = train_test_split(known_data["text"], known_data["class"], test_size=0.33, random_state=42)
    known_train = tfidf_2.fit_transform(known_train)
    known_test = tfidf_2.transform(known_test)

    # Cosine similarity 餘弦相似性 
    socres_novelty = cosine_similarity(novelty, Seen)
    socres_normality = cosine_similarity(known_test, known_train)

    return socres_novelty,socres_normality