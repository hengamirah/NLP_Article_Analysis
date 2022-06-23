# NLP_Article_Analysis
 Articles dataset sorted into 5 categories namely Sport, Tech, Business, Entertainment and Politics using Natural Language Processing Approach.

# PROBLEM STATEMENT
Provided text documents that has 5 categories, can we categorize unseen in articles into 5 categories?


# DISCUSSION
1) Model accuracy by only using 2 LSTM layer will only achieve 29% accuracy
2) Model accuracy using 1 embedding layer increases performance very minimally
3) By Adding masking, epoch increases the accuracy up to 82%
4) The model achieved high with average 83% F1-score and accuracy score of 83%
5) Model evaluated with test data has 83% accuracy 
6) when put earlystopping the model reduces accuracy to 29% only
7) To further increase the performance of NLP model: 
    #1) Increasing number of epochs, 
    #2) Increase number of samples
    #3) Change dropout rate value
    #4) Add word2vec to remove stop words from dataset

