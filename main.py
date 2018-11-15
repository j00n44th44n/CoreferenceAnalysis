# coded by jonathan & carlos C511
import sklearn
from sklearn.naive_bayes import BernoulliNB
import random
import nltk
import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names
from sklearn.externals import joblib
from os.path import exists

# languages
ES = 'spanish'
EN = 'english'
# grammars
# simple_en_grammar = r"""
#   NP: {(<PRP>)|(<DT>?<JJ>*<NN.*>+<POS><JJ>*<NN.*>+)|(<DT|PRP\$>?<JJ>*<NN.*>+)}     # chunk determiner/possessive, adjectives and noun
# """

simple_en_grammar = r"""
  NP: {(<PRP>)|(<DT>?<JJ>*<NN.*>+((<CC|IN>)<DT>?<JJ>*<NN.*>+)*)|(<DT>?<JJ>*<NN.*>+<POS><JJ>*<NN.*>+)|(<DT|PRP\$>?<JJ>*<NN.*>+)}
"""

en_grammar = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  PP: {<IN><NP>}               # Chunk prepositions followed by NP
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
  CLAUSE: {<NP><VP>}           # Chunk NP, VP
  """

np_en_grammar = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  CLAUSE: {<NP>*}              # Chunk all NP * don't know if thats what i want
  """
# getting current path
directory = os.getcwd()
# for steming use wnl.lematize('word')
wnl = WordNetLemmatizer()
# List of original_corpus
corpus = []
# In each index there is a list of all the sentences from ith-corpus
sentences_corpus = []
# t_c[i][j][k] => k-th word from j-th sentence from i-th corpus
token_corpus = []
# change every token for the morphological
morpho_token_corpus = []
# token pos tagged
pos_tag_token = []
# noun phrase sentences
noun_phrase_sentences = []
# Entity Recognition in Sentences 
entities_in_sentence = []
# # clasificador de nombres
# name_clasifier = None
# gender clasifier
m_gender = ['he','him','his','himself','Mr']
f_gender = ['she','her','hers','herself','Mrs']
# number clasifier 
s_number = ['i', 'you','he','she','it','me','him','her','who','mine','your','his','hers','this','that',
            'whom', 'which', 'what', 'whose', 'whoever', 'whatever', 'whichever', 'whomever','myself', 
            'yourself', 'himself', 'herself', 'itself','another','each','anything','nobody','nothing',
            'no one','none','other','anyone','somebody','someone','something','anybody','one','such',
            'more','little','my']
p_number = ['we','us','them','they','yours','ours','theirs','these','those','ourselves', 'themselves',
            'each other','one another','everybody','few','many','some','all','any','everyone','everything',
            'all','several','others','both','either','neither','much','enough']


# list of ith corpus
np_dict = []

# ! Task 0: load corpus... ok!
def gender_features(word):
    return {'last_letter': word[-1]}

def load_corpus():
    for filename in os.listdir(directory+"\\corpus"):
        with open(directory+"\\corpus\\"+filename, "r", encoding='utf8', errors='ignore') as f:
            text = f.read()
            corpus.append(text)
    # if not exists('name_clasifier.joblib'):
    #     labeled_names = ([(name, 'Male') for name in names.words('male.txt')] + [(name, 'Female') for name in names.words('female.txt')])
    #     random.shuffle(labeled_names)
    #     featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
    #     train_set = featuresets
    #     classifier = nltk.NaiveBayesClassifier.train(train_set)
    #     joblib.dump(classifier,'name_clasifier.joblib')
    #     name_clasifier = classifier
    # else:
    #     name_clasifier = joblib.load('name_clasifier.joblib')


# ! Task 1: Tokenization and sentences segmentation... ok!
def tokenization_and_sentence_segmentation():
    for i in range(len(corpus)):
        sentences_corpus.append(nltk.sent_tokenize(corpus[i],language=EN))
        tokenize_sentence = []
        for j in range(len(sentences_corpus[i])):
            tokenize_sentence.append(nltk.word_tokenize(sentences_corpus[i][j],language=EN))
        token_corpus.append(tokenize_sentence)

# ! Task 2: Morphological Processing... ok!
def morphological_processing():
    for i in range(len(token_corpus)):
        sentence = []
        for j in range(len(token_corpus[i])):
            word = []
            for k in range(len(token_corpus[i][j])):
                # word.append(wnl.lemmatize(token_corpus[i][j][k]))
                word.append(token_corpus[i][j][k])
            sentence.append(word)
        morpho_token_corpus.append(sentence)  

# ! Task 3: Postagger... ok!
def post_tagger():
    for i in range(len(morpho_token_corpus)):
        sentence = []     
        for j in range(len(morpho_token_corpus[i])):
            sentence.append(nltk.pos_tag(morpho_token_corpus[i][j],lang='eng'))
        pos_tag_token.append(sentence)

# ! Task 4: Noun Phrase Identification... ok!
def noun_phrase_identification():
    cp = nltk.RegexpParser(simple_en_grammar,loop=2)
    for i in range(len(pos_tag_token)):
        sentences_NP = []
        for j in range(len(pos_tag_token[i])):
            sentence = []
            tree = cp.parse(pos_tag_token[i][j])
            for subtrees in tree.subtrees():
                if subtrees.label() == 'NP':
                    sentence.append(subtrees)
                    sentence.extend(find_pos_pron(subtrees))
            sentences_NP.append(sentence)
        noun_phrase_sentences.append(sentences_NP)

# ! Task 6: Nested Noun Phrase Extraction... ok!
def find_pos_pron(tree):
    l = []
    j = 0
    for i in range(len(tree)):
        if tree[i][1] == 'PRP$':
            n_tree = nltk.Tree('NP',[(tree[i][0],tree[i][1])])
            l.append(n_tree)
            j = i+1
            continue
        if tree[i][1] == 'POS' and i>0:
            n_tree = nltk.Tree('NP',[(tree[i-1][0],tree[i-1][1])])
            l.append(n_tree)
            j = 1+1
            continue
        if (tree[i][1] == 'CC' or tree[i][1] == 'IN'):
            n_tree = nltk.Tree('NP',[(tree[j:i])])
            n_tree2 = nltk.Tree('NP',[(tree[i+1:])])
            l.append(n_tree)
            l.append(n_tree2)
            j = i+1
            continue
    return l

# ! Task 5: Named Entity Recognition... ok!
def named_entity_recognition():
    for i in range(len(pos_tag_token)):
        sentence_entity = []
        for j in range(len(pos_tag_token[i])):
            tree = nltk.ne_chunk(pos_tag_token[i][j])
            for subtrees in tree.subtrees():
                if subtrees.label() == 'GPE' or subtrees.label() == 'PERSON' or subtrees.label() == 'ORGANIZATION':
                    sentence_entity.append(subtrees)
        entities_in_sentence.append(sentence_entity)

def get_np_info(tree,i,j,k,name_clasifier):
    nn = ''
    clase = 'unknow'
    tipo = 'unknow'
    gender = 'unknow'
    number = 'unknow'
    for i in range(len(tree)):
        if tree[i][1] == 'NNP': # ? proper name
            text = ''
            _tree = nltk.ne_chunk(noun_phrase_sentences[i][j][k])
            while i < len(tree) and tree[i][1] == 'NNP':
                text += ' ' + tree[i][0]                
                for subtrees in _tree.subtrees():
                    if subtrees.label() == 'PERSON':
                        clase = 'PERSON'
                        if gender == 'unknow':
                            gender = name_clasifier.classify(gender_features(tree[i][0]))
                    if subtrees.label() == 'GPE':
                        clase = 'Object'
                        tipo = 'GPE'
                    if subtrees.label() == 'ORGANIZATION':
                        clase = 'Object'
                        tipo = 'Organization'
                # TODO clasificalo y asigna clase, tipo, genero
                i+=1
            nn += ' ' + text
            i-=1
            continue
        if tree[i][1] == 'NN':
            text = ''
            while i < len(tree) and tree[i][1] == 'NN':
                text += ' ' + tree[i][0]
                clase = 'unknow'
                tipo = 'unknow'
                gender = 'unknow'
                i+=1
            nn += ' ' + text
            i-=1
        if tree[i][1] == 'DT' or tree[i][1] == 'PRP' or tree[i][1] == 'POS':
            nn += ' ' + tree[i][0]
            if tree[i][0] in m_gender:
                clase = 'Person'
                gender = 'Male'
                number = 'singular'
                continue
            if tree[i][0] in f_gender:
                clase = 'Person'
                gender = 'Female'
                number = 'singular'
                continue
            if tree[i][0] in s_number:
                number = 'singular'
                continue
            if tree[i][0] in p_number:
                number = 'plural'
    return nn[1:],clase,tipo,gender,number

# ! Task 7: Semantic Class Determination
# clases are:
# ? Person/Animal (male,female,unknow)
# ? Object (Organization,Localization)
def semantic_class_determination():
    labeled_names = ([(name, 'Male') for name in names.words('male.txt')] + [(name, 'Female') for name in names.words('female.txt')])
    random.shuffle(labeled_names)
    featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
    train_set = featuresets
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    for i in range(len(noun_phrase_sentences)):
        d = {}
        for j in range(len(noun_phrase_sentences[i])):
            for k in range(len(noun_phrase_sentences[i][j])):
                snn,clase,tipo,gender,number = get_np_info(noun_phrase_sentences[i][j][k],i,j,k,classifier)
                nn = str(noun_phrase_sentences[i][j][k])
                if not nn in d:
                    d[nn] = (clase,tipo,gender,number)
                else:
                    if clase == 'unknow' and d[nn][0] != 'unknow':
                        clase = d[nn][0]
                    if tipo == 'unknow' and d[nn][1] != 'unknow':
                        tipo = d[nn][0]
                    if gender == 'unknow' and d[nn][2] != 'unknow':
                        gender = d[nn][0]
                    if number == 'unknow' and d[nn][3] != 'unknow':
                        number = d[nn][0]
                    d[nn] = (clase,tipo,gender,number)
        np_dict.append(d)


# ! Task 8: Get Markables()
def get_markables():
    pass

def main():
    load_corpus()
    tokenization_and_sentence_segmentation()
    morphological_processing()
    post_tagger()
    noun_phrase_identification()
    named_entity_recognition()
    semantic_class_determination()
    # get_markables()
    print(str(corpus))
    # print(str(token_corpus))
    # print(str(morpho_token_corpus))
    print(str(pos_tag_token))
    print(str(noun_phrase_sentences))
    # print(str(entities_in_sentence))
main()