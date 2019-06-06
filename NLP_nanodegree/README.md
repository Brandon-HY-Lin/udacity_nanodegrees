# Natural Language Processing (NLP) Nanodegree

## Projects
1. [Part of Speech Tagging](https://github.com/Brandon-HY-Lin/hmm-tagger)
	- Purpose: Tag verb, noun, etc. in sentences.
	- Library: Pomegranate.
	- Algorithm: HMM and Supervised Learning.
	- [Main Program](https://github.com/Brandon-HY-Lin/hmm-tagger/blob/master/HMM%20Tagger.ipynb)

2. [Machine Translation](https://github.com/Brandon-HY-Lin/aind2-nlp-capstone)
	- Purpose: Translate English texts to French texts.
	- Framework: Keras.
	- Algorithm: Recursive Encoder-Decoder RNN.
	- Dataset: Subset of [WMT](http://www.statmt.org/)
	- [Report](https://github.com/Brandon-HY-Lin/aind2-nlp-capstone/blob/master/Report.md)
	- [Main Program](https://github.com/Brandon-HY-Lin/aind2-nlp-capstone/blob/master/machine_translation.ipynb)

3. [DNN Speech Recognizer](https://github.com/Brandon-HY-Lin/AIND-VUI-Capstone)
	- Purpose: Implement UVI (User-Voice-Interface).
	- Framework: Keras
	- Algorithm: 2-Dimensional CNN + RNN + Dense Layer.
	- Dataset: [LibriSpeech](http://www.openslr.org/12/)
	- [Report](https://github.com/Brandon-HY-Lin/AIND-VUI-Capstone/blob/master/Report.md)
	- [Main Program](https://github.com/Brandon-HY-Lin/AIND-VUI-Capstone/blob/master/vui_notebook.ipynb)

## Labs
* Part 1: Introduction to Natural Language
	* [Text Processing](https://github.com/Brandon-HY-Lin/AIND-NLP/blob/master/text_processing.ipynb)
		- Purpose: Tokenize articles
		- Libraries: Pandas and NLTK
		- Key APIs:
			- Tokenize: nltk.tokenize.word_tokenize(text)
			- Stopwords: nltk.corpus.stopwords.words('english')
			- Stem/Lemmatize:
				- Stem: nltk.stem.PorterStemmer().stem(word)
				- Lemmatize: nltk.stem.WordNetLemmatizer().lemmatize(word, pos='v')

	* [Spam Classifier](https://github.com/Brandon-HY-Lin/NLP-Exercises/blob/master/1.5-spam-classifier/Bayesian_Inference.ipynb)
		- Purpose: Classify spam email.
		- Libraries: Pandas and Scikit-Learn.
		- Algorithm: Apply naive Bayes to BOW (Bag of Words).
		- Key Concept:
			- Bag Of Words: It is a statictis of corpus and ingnores the order of words. For example, "chicago bulls" might be treated as a city and an animal, rather than the basketball team.
		- Key APIs:
			- Pre-process + Vectorize + BOW: sklearn.feature_extraction.text.CountVectorizer().fit_transform(text)
			- Split train/test set: sklearn.cross_validation.train_test_split()
			- Naive Bayes: sklearn.naive_bayes.MultinomialNB().fit()
			- F1 score, recall score, ...:
				- sklearn.metrics.f1_score()
				- sklearn.metrics.accuracy_score()
				- sklearn.metrics.precision_score()
				- sklearn.metrics.recall_score()

	* [IBM Bookworm](https://github.com/Brandon-HY-Lin/AIND-NLP-Bookworm)
		- Purpose: A simple question-answering system built using IBM Watson's NLP services.

* Part 2: Computing with Natural Language
	* [Topic Modeling](https://github.com/Brandon-HY-Lin/NLP-Exercises/blob/master/2.2-topic-modeling/Latent_dirichlet_allocation.ipynb)
		- Purpose: Classify text to a particular topic
		- Libraries: Gensim and Pandas.
		- Algorithm: LDA (Latent Dirichlet Allocation) using TF-IDF (Trem Frequency-Inverse Document Frequency).
		- Key concept:
			- TF-IDF: Consider a document containing 100 words wherein the word 'tiger' appears 3 times.
				- TF:
					- The term frequency (i.e., tf) for 'tiger' is then: TF = (3 / 100) = 0.03.
				- IDF:
					- Now, assume we have 10 million documents and the word 'tiger' appears in 1000 of these. Then, the inverse document frequency (i.e., idf) is calculated as: IDF = log(10,000,000 / 1,000) = 4.
				- TF-IDF:
					- Thus, the Tf-idf weight is the product of these quantities: TF-IDF = 0.03 * 4 = 0.12.
		- Key APIs:
			- Normalize and Tokenize: gensim.utils.simple_preprocess(text)
			- Stopswords: gensim.parsing.preprocessing.STOPWORDS
			- Lemmatize/Stem:
				- Lemmatize: nltk.stem.WordNetLemmatizer().lemmatize(word)
				- Stem: nltk.stem.SnowballStemmer().stem(word)
			- Create Dictionary: gensim.corpora.Dictionary(docs)
			- BOW/TF-IDF:
				- BOW: bow_corpus = gensim.corpora.Dictionary(docs).doc2bow(text)
				- TF-IDF: tfidf_corpus = gensim.models.TfidfModel(bow_corpus)
			- LDA: 
				- gensim.models.LdaMulticore(bow_corpus, num_topics)
				- gensim.models.LdaMulticore(tfidf_corpus, num_topics)
				
	* [Sentiment Analysis](https://github.com/Brandon-HY-Lin/NLP-Exercises/blob/master/2.3-sentiment-analysis/sentiment_analysis_udacity_workspace.ipynb)
		- Purpose: Predict positive or negative sentiment upon a comment.
		- Libraries: Sklearn.
		- Algorithm: Naive Bayes and Gradient-Boosted Decision Tree classifier.

	* [Attention Basic](https://github.com/Brandon-HY-Lin/NLP-Exercises/blob/master/2.5-attention/Attention%20Basics.ipynb)
		- Purpose: Implement basic block in Attention algorithm.
		- Algorithm: Attention

	* [RNN Keras Lab](https://github.com/Brandon-HY-Lin/NLP-Exercises/blob/master/2.6-rnn-keras-lab/Deciphering%20Code%20with%20Character-Level%20RNN.ipynb)
		- Purpose: Decipher strings encrypted with a certain cipher.
		- Framework: Keras.
		- Algorithm: GRU.

* Part 3: Communicating with Natural Language
	* [Voice Data](https://github.com/Brandon-HY-Lin/AIND-VUI-Lab-Voice-Data)
		- Purpose: Explore the LibriSpeech data set and format
