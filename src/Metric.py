def NoveltyScore(socres_novelty,socres_normality,methods):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    if methods == "AUTO":
        nor_max_scale = []
        nor_std = []
        for i in socres_normality:
            nor_max_scale.append((i - min(socres_novelty+socres_normality))/(max(socres_novelty+socres_normality)-min(socres_novelty+socres_normality)))
            nor_std.append(np.std(i))
    
    
        nov_max_scale = []
        nov_std = []
        for i in socres_novelty:
            nov_max_scale.append((i - min(socres_novelty+socres_normality))/(max(socres_normality+socres_novelty)-min(socres_normality+socres_novelty)))
            nov_std.append(np.std(i))
            
        
    if methods == "BERT":
        nor_max = []
        nor_std = []
        for i in socres_normality:
            nor_max.append(1 - max(i))
            nor_std.append(np.std(i))
    
        nov_max = []
        nov_std = []
        for i in socres_novelty:
            nov_max.append(1 - max(i))
            nov_std.append(np.std(i))
    
        nor_max_scale = []
        for i in nor_max:
            nor_max_scale.append((i - min(nov_max+nor_max))/(max(nov_max+nor_max)-min(nov_max+nor_max)))
    
    
        nov_max_scale = []
        for i in nov_max:
            nov_max_scale.append((i - min(nov_max+nor_max))/(max(nov_max+nor_max)-min(nov_max+nor_max)))
            
    if methods == "TFIDF":
        nor_max_scale = []
        nor_std = []
        for i in socres_normality:
            nor_max_scale.append(1 - max(i))
            nor_std.append(np.std(i))
    
        nov_max_scale = []
        nov_std = []
        for i in socres_novelty:
            nov_max_scale.append(1 - max(i))
            nov_std.append(np.std(i))
        
    
    nov_max_describe = pd.DataFrame(nov_max_scale)
    nor_max_describe = pd.DataFrame(nor_max_scale)
    nov_std_describe = pd.DataFrame(nov_std)
    nor_std_describe = pd.DataFrame(nor_std)
    
    # Plot comparison graph
    plt.title(f"Distribution of {methods} novelty score")
    plt.hist(nov_max_describe,bins =50, alpha=0.5,color ="red", label='Novelty')
    plt.hist(nor_max_describe,bins =50, alpha=0.5,color ="green", label='Normality')
    plt.legend(loc='upper right')
    plt.xlabel("score")
    plt.ylabel("number")
    plt.show()


    Novelty = pd.DataFrame({
                    'Max':nov_max_describe[0].tolist(),
                    'Std':nov_std_describe[0].tolist(),                  
                    'Target':1})
    Normality = pd.DataFrame({
                    'Max':nor_max_describe[0].tolist(),
                    'Std':nor_std_describe[0].tolist(),
                    'Target':0})
    return Novelty,Normality


# # Correlation
def Correlation(X,y):
    import scipy

    pearsons_coefficient = scipy.stats.pearsonr(X, y)[0]
    pvalue = scipy.stats.pearsonr(X, y)[1]
    print("The pearson's coeffient of the x and y inputs are: \n" ,pearsons_coefficient)
    print("P-value is : \n",pvalue," < 0.0001", " ==> extremely significant")

def jaccard_coefficient(A, B):
    #Find intersection of two sets
    nominator = A.intersection(B)

    #Find union of two sets
    denominator = A.union(B)

    #Take the ratio of sizes
    similarity = len(nominator)/len(denominator)
    
    return similarity

def continuous_jaccard_coefficient(A,B):
    size_A = max(A) - min(A)
    size_B = max(B) - min(B)
    r = size_B/size_A
    
    x = ((size_A-size_B)/2+(size_A+size_B)/2)/2
    index = ((size_A**2)*r*(1+r)-2*r*size_A*x)/(2*(size_A**2)*(1+r**2)-(size_A**2)*r*(1+r)+2*r*size_A*x)
    return index

def Overlap(X,y):
    # Measure the Overlap 

    from py_stringmatching.similarity_measure.overlap_coefficient import OverlapCoefficient
    oc = OverlapCoefficient()
    OverlapCoefficient = oc.get_sim_score(X.to_list(),y.to_list())
    jaccardCoefficient = jaccard_coefficient(set(X), set(y))
    continuousJaccard = continuous_jaccard_coefficient(X.to_list(),y.to_list())
    print("Overlap Coefficient :",OverlapCoefficient )
    
    print("Jaccard Coefficient :",jaccardCoefficient)
    print("Jaccard Coefficient (continuous version) :",continuousJaccard)
