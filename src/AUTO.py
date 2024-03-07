def AutoEncoder(known_data,novelty_data):   
    from sklearn.model_selection import train_test_split
    from sentence_transformers import SentenceTransformer
    
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.backend import clear_session
    
    
    # Embedding
    known_x_train, known_x_test, known_y_train, known_y_test  = train_test_split(known_data["text"], known_data["class"], test_size=(novelty_data.shape[0]/(novelty_data.shape[0]+known_data.shape[0])), random_state=42)
    bertModel = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    
    X_train = bertModel.encode(list(known_x_train))
    novelty = bertModel.encode(list(novelty_data["text"]))
    normal = bertModel.encode(list(known_x_test))
    
    # Training
    clear_session()
    in_layer = Input(X_train.shape[1])
    encoder = Dense(256, activation="relu")(in_layer)
    encoder = Dense(128, activation="relu")(encoder)
    encoder = Dense(64, activation="relu")(encoder)
    decoder = Dense(128, activation="relu")(encoder)
    decoder = Dense(256, activation="relu")(decoder)
    decoder = Dense(X_train.shape[1], activation="sigmoid")(decoder)


    #
    autoencoder = Model(in_layer, decoder)
    encoder = Model(in_layer, encoder)
    autoencoder.compile("adam", "mean_squared_logarithmic_error")
    autoencoder.fit(X_train, X_train, epochs=100, batch_size=16)



    reconstruction = autoencoder.predict(novelty)
    novelty_loss = tf.keras.losses.msle(reconstruction,novelty)
    reconstruction = autoencoder.predict(normal)
    normal_loss = tf.keras.losses.msle(reconstruction,normal)
    
    return np.array(novelty_loss).tolist(),np.array(normal_loss).tolist()