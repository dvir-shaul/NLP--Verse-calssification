# NLP--Verse-calssification
Final report - Word embedding
## Main goal for this article is to implement word embedding technique on a verse that has been taken from the "Torah" and try a good classifying method on them that will predict the book it was taken from, in this article will be implemented and used Keras Sequential model to try and predict our verse, in hopping to beat the last task prediction record that used Neural network method for prediction the same data.

### Introduction:
Word embedding is one of the most popular representation of document vocabulary. It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc.
Main problem in this article is to classifying verses from the "Torah" to the book it was taken from, "Torah" includes 5 different books: Genesis, Exodus, Leviticus, Numbers, Deuteronomy, which mean the program will have to try and predict 5 different classes for each verse.
In this essay we will focus about natural language processing (NLP) with solution using word embedding, which is a term used for the representation of words for text analysis, typically in the form of a real-valued vector that encodes the meaning of the word such that the words that are closer in the vector space are expected to be similar in meaning. Mathematically, the cosine of the angle between such vectors should be close to 1, i.e., angle close to 0.
