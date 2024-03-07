# Source Code Introduction

Here's is the code for a pipeline that performs patent novelty analysis using different methods and evaluation metrics. The pipeline consists of several modules, each responsible for a specific task. Below is an overview of each module and its purpose.

### Main.py
This is the main script that orchestrates the entire pipeline. It imports and calls the necessary functions from the other modules to execute the patent novelty analysis workflow.

### DataPreprocessing.py
This module handles the data preprocessing tasks. It includes functions for text cleaning and merging multiple pandas datasets. The data preprocessing ensures that the input data is in a suitable format for further analysis.

### Method 1: TFIDF_Sim.py
This module implements the first method for calculating patent novelty using TF-IDF (Term Frequency-Inverse Document Frequency). It includes functions for computing TF-IDF scores and measuring the similarity between patents based on these scores.

### Method 2: BERT_Sim.py
This module implements the second method for calculating patent novelty using Sentence-BERT. It includes functions for fine-tuning BERT models on patent data and computing similarity scores between patents.

### Method 3: AUTO.py
This module implements the third method for calculating patent novelty using an automated approach. 

### Metric.py
This module handles the evaluation metric calculations. It includes functions for computing the novelty score, correlation coefficient, Jaccard index, and overlap index. These metrics provide insights into the novelty and similarity of the patents.

### BERTFineTune.py
This module focuses on fine-tuning the Sentence-BERT model for patent data. It includes functions for preprocessing the text, creating Sentence-BERT input embeddings, and training the BERT model on the patent dataset.

### stockMarketData.py
This module is responsible for scraping stock market data. It includes functions for retrieving stock market data from a reliable source. This data is used for analysis and correlation with the patent novelty results.

Please note that this module requires the file "Startup_Listed_In_NASDAD_2000-2020.xlsx" to be present in the same directory.

### PatentNovelty.py
This module generates the final patent novelty results. It includes functions for processing the "all_file.json" file and producing the novelty scores for each patent. The output of this module represents the novelty of each patent based on the selected method and evaluation metrics.

Please note that this module requires the file "all_file.json" to be present in the same directory.

Feel free to explore the code in this repository and adapt it to your specific needs. If you have any questions or face any issues, please don't hesitate to reach out to the repository owner.

