import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger


stemmer = PorterStemmer()
tagger = PerceptronTagger()
lemmatizer = WordNetLemmatizer()


def remove_stop_words(str):
    stop_words = set(stopwords.words("english"))
    words = str.split()
    clean_str = " ".join([word for word in words if not word in stop_words])
    return clean_str


def remove_features(str):

    url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
    ampersand_re = re.compile("&amp;")
    num_re = re.compile('(\\d+)')
    mention_re = re.compile('@(\w+)')
    alpha_num_re = re.compile("^[a-z0-9_.]+$")

    # convert to lowercase
    str = str.lower()
    # remove hyperlinks
    str = url_re.sub('', str)
    # remove @mentions
    str = mention_re.sub('', str)
    # remove & symbols
    str = ampersand_re.sub('', str)
    # remove puncuation
    str = punc_re.sub('', str)
    # remove numeric 'words'
    str = num_re.sub('', str)

    words = str.split()

    clean_str = " ".join(
        [word for word in words if alpha_num_re.match(word) and len(word) > 2])

    return clean_str


def pos_tagging(str):
    clean_str = []
    # noun tags
    nn_tags = ['NN', 'NNP', 'NNP', 'NNPS', 'NNS']
    # adjectives
    jj_tags = ['JJ', 'JJR', 'JJS']
    # verbs
    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    nltk_tags = nn_tags + jj_tags + vb_tags

    # break string into 'words'
    words = str.split()

    # tag the text and keep only those with the right tags
    tagged_text = tagger.tag(words)
    for tagged_word in tagged_text:
        if tagged_word[1] in nltk_tags:
            clean_str.append(tagged_word[0])

    return " ".join(clean_str)


def lemmatize(str):
    words = str.split()
    tagged_words = tagger.tag(words)
    clean_str = []

    for word in tagged_words:
        if 'n' in word[1].lower():
            lemma = lemmatizer.lemmatize(word[0], pos='n')
        else:
            lemma = lemmatizer.lemmatize(word[0], pos='v')

        clean_str.append(lemma)

    return " ".join(clean_str)


def preproc_pipeline(str):
    rm_stop = remove_stop_words(str)
    rm_feat = remove_features(rm_stop)
    rm_pos = pos_tagging(rm_feat)
    lem = lemmatize(rm_pos)
    return lem
