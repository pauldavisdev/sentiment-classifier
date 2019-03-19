from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import re, string

class Preprocessor:

    stemmer = PorterStemmer()

    stopwords_english = stopwords.words('english')

    emoticons = [
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3', ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ]

    def clean_texts(self, trainX, testX):
        
        trainX_cleaned = []
        
        testX_cleaned = []

        for text in trainX:
            trainX_cleaned.append(self.clean_tweet(text))

        for text in testX:
            testX_cleaned.append(self.clean_tweet(text))

        return trainX_cleaned, testX_cleaned

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

        # tokenize tweets
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

        tweet_tokens = tokenizer.tokenize(tweet)

        cleaned_tweet = []

        for word in tweet_tokens:
            # remove punctuation
            word = re.sub(re.compile('[%s]' % re.escape(string.punctuation)), '', word)
            if (word not in self.stopwords_english and
                word not in self.emoticons):
                stemmed_word = self.stemmer.stem(word)
                cleaned_tweet.append(stemmed_word)
        
        return cleaned_tweet
