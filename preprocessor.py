from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re, string

class Preprocessor:

    stemmer = PorterStemmer()

    stopwords_english = stopwords.words('english')

    def clean_texts(self, trainX):
        
        trainX_cleaned = []

        for text in trainX:
            trainX_cleaned.append(self.clean_tweet(text))

        return trainX_cleaned

    def clean_tweet(self, tweet):

        # remove stock market tickers like $GE
        tweet = re.sub(r'\$\w*', '', tweet)
    
        # remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)
    
        # remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        
        # remove hashtags
        # only removing the hash # sign from the word
        tweet = re.sub(r'#', '', tweet)

        # remove numbers
        tweet = re.sub(r'\d+', '', tweet)

        # remove usernames and set all to lowercase
        # reduce length of 4 or more consecutive characters in a word to 3
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

        tweet_tokens = tokenizer.tokenize(tweet)

        stemmed_tweet = []

        for word in tweet_tokens:
            #word = self.stemmer.stem(word)
            stemmed_tweet.append(word)
        return ' '.join(stemmed_tweet)
