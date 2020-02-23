# momconnect

Since 2014 MomConnect has provided healthcare information and emotional support in all 11 official languages of South Africa to over 2.6 million pregnant and breastfeeding women, via SMS and WhatsApp. 
However, the service has struggled to scale efficiently with the growing user base and increase in incoming questions, resulting in a current median response time of 20 hours. 
The aim of our study is to investigate the feasibility of automating the manual answering process. 
This study consists of two parts: i) answer selection, a form of information retrieval, and ii) natural language processing (NLP), where computers are taught to interpret human language. 
Our problem is unique in the NLP space, as we work with a closed-domain question-answering dataset, with questions in 11 languages, many of which are low-resource, with English template answers, unreliable language labels, code-mixing, shorthand, typos, spelling errors and inconsistencies in the answering process. 
The shared English template answers and code-mixing in the questions can be used as cross-lingual signals to learn cross-lingual embedding spaces. 
We combine these embeddings with various machine learning models to perform answer selection, and find that the Transformer architecture performs best, achieving a top-1 test accuracy of $61.75\%$ and a top-5 test accuracy of $91.16\%$. 
It also exhibits improved performance on low-resource languages when compared to the long short-term memory (LSTM) networks investigated. 
Additionally, we evaluate the quality of the cross-lingual embeddings using parallel English-Zulu question pairs, obtained using Google Translate. 
Here we show that the Transformer model produces embeddings of parallel questions that are very close to one another, as measured using cosine distance. This indicates that the shared template answer serves as an effective cross-lingual signal, and demonstrates that our method is capable of producing high quality cross-lingual embeddings for low-resource languages like Zulu. Further, the experimental results demonstrate that automation using a top-5 recommendation system is feasible.

