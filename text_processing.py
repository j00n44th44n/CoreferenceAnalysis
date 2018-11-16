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
  NP: {(<PRP|PRP\$>)|(<DT>?<JJ>*<NN.*>+((<CC|IN>)<DT>?<JJ>*<NN.*>+)*)|(<DT>?<JJ>*<NN.*>+<POS><JJ>*<NN.*>+)|(<DT|PRP\$>?<JJ>*<NN.*>+)}
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
# token pos tagged
pos_tag_token = []
# noun phrase sentences
noun_phrase_sentences = []
# # clasificador de nombres
# name_clasifier = None
# gender clasifier
p_form = ['mr', 'mister','mrs','dr', 'miss', 'ms', 'mistress','dra']
m_gender = ['he','him','his','himself','mr', 'mister']
f_gender = ['she','her','hers','herself','mrs', 'miss', 'ms', 'mistress']
# number clasifier 
s_number = ['i', 'you','he','she','it','me','him','her','who','mine','your','his','hers','this','that',
            'whom', 'which', 'what', 'whose', 'whoever', 'whatever', 'whichever', 'whomever','myself', 
            'yourself', 'himself', 'herself', 'itself','another','each','anything','nobody','nothing',
            'no one','none','other','anyone','somebody','someone','something','anybody','one','such',
            'more','little','my','our']
p_number = ['we','us','them','they','yours','ours','theirs','these','those','ourselves', 'themselves',
            'each other','one another','everybody','few','many','some','all','any','everyone','everything',
            'all','several','others','both','either','neither','much','enough']

true  = 1
false  = 0
unknown  = 2


# list of ith document
np_dict = []
# list of all NP from document ith
# ? np_lists[i][j] jth np del doc ith
np_lists = []

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

# ! Task 3: Postagger... ok!
def post_tagger():
    for i in range(len(token_corpus)):
        sentence = []     
        for j in range(len(token_corpus[i])):
            sentence.append(nltk.pos_tag(token_corpus[i][j],lang='eng'))
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
            n_tree = nltk.Tree('NP',(tree[j:i]))
            n_tree2 = nltk.Tree('NP',(tree[i+1:]))
            l.append(n_tree)
            l.append(n_tree2)
            j = i+1
            continue
    return l


def get_np_info(tree,a,b,c,name_clasifier):
    nn = ''
    clase = 'unknown'
    tipo = 'unknown'
    gender = 'unknown'
    number = 'unknown'
    i = 0
    while i < len(tree):
        if tree[i][1] == 'NNP' or tree[i][1] == 'NNPS': # ? proper name
            text = ''
            j = i                   
            name = ''
            while i < len(tree) and (tree[i][1] == 'NNP' or tree[i][1] == 'NNPS'):
                if (tree[i][0]).lower() in p_form:
                    clase = 'PERSON'
                    number = 'singular'
                    if (tree[i][0]).lower() in p_form[:4]:
                        gender = 'Male'
                    else: 
                        gender = 'female'
                text += ' ' + tree[i][0]
                i+=1
            if clase == 'PERSON':
                continue
            _tree = nltk.ne_chunk(noun_phrase_sentences[a][b][c][j:i])            
            for subtrees in _tree.subtrees():
                if subtrees.label() == 'PERSON':
                    if clase == 'unknown':
                        clase = 'PERSON'
                    if gender == 'unknown':
                        gender = name_clasifier.classify(gender_features(tree[j][0]))
                    if number == 'unknown':
                        number = 'singular'
                    continue
                if subtrees.label() == 'GPE':
                    if clase == 'unknown':
                        clase = 'Object'
                    if tipo == 'unknown':
                        tipo = 'GPE'
                    continue
                if subtrees.label() == 'ORGANIZATION':
                    if clase == 'unknown':
                        clase = 'Object'
                    if tipo == 'unknown':
                        tipo = 'Organization'
                # TODO clasificalo y asigna clase, tipo, genero
            nn += text
            continue
        elif tree[i][1] == 'NN':
            text = ''
            while i < len(tree) and (tree[i][1] == 'NN' or tree[i][1] == 'NNS'):
                text += ' ' + tree[i][0]
                # clase = 'unknown'
                # tipo = 'unknown'
                # gender = 'unknown'
                if number == 'unknown':
                    number = 'singular'                
                i+=1
            nn += ' ' + text
        elif tree[i][1] == 'NNS':
            text = ''
            while i < len(tree) and (tree[i][1] == 'NNS' or tree[i][1] == 'NN'):
                text += ' ' + tree[i][0]
                # clase = 'unknown'
                # tipo = 'unknown'
                # gender = 'unknown'
                if number == 'unknown':
                    number = 'plural'
                i+=1
            nn += ' ' + text
        elif tree[i][1] == 'DT' or tree[i][1] == 'PRP' or tree[i][1] == 'PRP$' or tree[i][1] == 'POS':
            nn += ' ' + tree[i][0]
            if (tree[i][0]).lower() in m_gender:
                clase = 'Person'
                gender = 'Male'
                number = 'singular'
            if (tree[i][0]).lower() in f_gender:
                clase = 'Person'
                gender = 'Female'
                number = 'singular'
            if (tree[i][0]).lower() in s_number:
                number = 'singular'
            if (tree[i][0]).lower() in p_number:
                number = 'plural'
        elif tree[i][1] == 'CC':
            clase = 'unknown'
            tipo = 'unknown'
            gender = 'unknown'
            number = 'plural'
            break
        elif tree[i][1] == 'IN':
            break
        i+=1
    return nn[1:],clase,tipo,gender,number

# ! Task 7: Semantic Class Determination
# clases are:
# ? Person/Animal (male,female,unknown)
# ? Object (Organization,Localization)
def semantic_class_determination():
    labeled_names = ([(name, 'Male') for name in names.words('male.txt')] + [(name, 'Female') for name in names.words('female.txt')])
    random.shuffle(labeled_names)
    featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
    train_set = featuresets
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    for i in range(len(noun_phrase_sentences)):
        d = {}
        npl = []
        for j in range(len(noun_phrase_sentences[i])):
            for k in range(len(noun_phrase_sentences[i][j])):
                snn,clase,tipo,gender,number = get_np_info(noun_phrase_sentences[i][j][k],i,j,k,classifier)
                nn = str(noun_phrase_sentences[i][j][k])
                if not nn in d:
                    d[nn] = {
                                'class': clase,
                                'type': tipo,
                                'gender': gender,
                                'number': number,
                                'sentence': j,
                                'doc': i,
                            }
                    npl.append(noun_phrase_sentences[i][j][k])
                else:
                    if clase == 'unknown' and d[nn]['class'] != 'unknown':
                        clase = d[nn]['class']
                    if tipo == 'unknown' and d[nn]['type'] != 'unknown':
                        tipo = d[nn]['type']
                    if gender == 'unknown' and d[nn]['gender'] != 'unknown':
                        gender = d[nn]['gender']
                    if number == 'unknown' and d[nn]['number'] != 'unknown':
                        number = d[nn]['number']
                    oldj = d[nn]['sentence']
                    oldi = d[nn]['doc']
                    d[nn] = {
                                'class': clase,
                                'type': tipo,
                                'gender': gender,
                                'number': number,
                                'sentence': oldj,
                                'doc': oldi,
                            }
        np_lists.append(npl)
        np_dict.append(d)

def is_same_phrase (antecedent:nltk.Tree, anaphor:nltk.Tree):
    partial_antecedent = []
    for word, entity in antecedent.leaves():
        if not (entity == 'DT' or entity == 'JJ'):
            partial_antecedent.append(word)

    partial_anaphor = []
    for word, entity in anaphor.leaves():
        if not (entity == 'DT' or entity == 'JJ'):
            partial_anaphor.append(word)

    if len(partial_antecedent) == len(partial_anaphor):
        for word1, word2 in zip(partial_anaphor, partial_antecedent):
            if word1 != word2:
                return false           
        return true
    else:
        return false 

def is_proper_name(noun_phrase:nltk.Tree):
    for word, typ in noun_phrase.leaves():
        if (not word.istitle()) and (not typ in []):
            return False
    return True

def is_alias (antecedent_np:nltk.Tree, antecedent_dict, anaphor_np:nltk.Tree, anaphor_dict):
    if antecedent_dict['class'] != anaphor_dict['class']:
        return False

    if antecedent_dict['class'] == 'PERSON':
        return (antecedent_np.leaves())[-1][0] == (anaphor_np.leaves())[-1][0] and antecedent_dict['gender'] == anaphor_dict['gender']

    if antecedent_dict['class'] == 'ORGANIZATION':
        acronym, words = (antecedent_np.leaves()[0], anaphor_np.leaves()) if 1 == len(antecedent_np.leaves()) < len(anaphor_np.leaves()) else (anaphor_np.leaves()[0], antecedent_np.leaves())
        
        if len(acronym) != len(words):
            return False

        for i in len(acronym):
            if str(acronym[i]) != words[i][0]:
                return False

        return True 

    return False


# ! Task 8: Get Markables()
def get_markables():
    load_corpus()
    tokenization_and_sentence_segmentation()
    post_tagger()
    noun_phrase_identification()
    semantic_class_determination()    
    get_markables()

    pair_noun_phrases = []
    data = []
    
    doc_num = 0
    for doc in np_lists:
        for i in range(len(doc) - 1):
            antecedent_np = doc[i]
            antecedent_dict = np_dict[doc_num][str(doc[i])]
            for j in range(i + 1, len(doc )):
                anaphor_np = doc[j]
                anaphor_dict = np_dict[doc_num][str(doc[j])]

                features = []

                # 1 - distance feature (DIST)
                features.append(anaphor_dict['sentence'] - antecedent_dict['sentence'])

                # 2 - antecedent-pronoun feature (I-PRONOUN)
                features.append(true  if len(antecedent_np.leaves()) == 1 and 'PRP' in antecedent_np.leaves()[0][1] else false )

                # 3 - anaphor-pronoun feature (I-PRONOUN)
                features.append(true  if len(anaphor_np.leaves()) == 1 and 'PRP' in anaphor_np.leaves()[0][1] else false )

                # 4 - string match feature (STR-MATCH)
                features.append(is_same_phrase (antecedent_np, anaphor_np))
                
                # 5 - definite noun phrase feature (DEF-NP)
                features.append(true  if (anaphor_np.leaves()[0][0]).lower() == 'the' else false ) 
                
                # 6 - demonstrative noun phase feature (DEM-NP)
                features.append(true  if (anaphor_np.leaves()[0][0]).lower() in ['this', 'that', 'these', 'those']  else false ) 

                # 7 - number agreement feature (NUMBER)
                features.append(true if antecedent_dict['number'] == anaphor_dict['number'] else false)
                
                # 8 - semantic class agreement feature (SEMCLASS)  
                features.append(true if antecedent_dict['class'] == anaphor_dict['class'] else false)

                # 9 - gender agreement feature (GENDER)
                features.append(unknown if antecedent_dict['gender'] == 'unknown' or anaphor_dict['gender'] == 'unknown' \
                                        else true if antecedent_dict['gender'] == anaphor_dict['gender'] \
                                        else false)

                # 10 - both-proper-names feature (PROPER-NAME)
                features.append(true if is_proper_name(anaphor_np) == is_proper_name(antecedent_np) == True else false)

                # 11 - alias feature (ALIAS)
                features.append(true if is_alias(antecedent_np, antecedent_dict, anaphor_np, antecedent_dict) else false)

                # 12 - appositive feature (APPOSITIVE)
                features.append(0)       

                str1 = ''
                for word in antecedent_np.leaves():
                    str1 += (word[0] + ' ')
                str2 = ''
                for word in anaphor_np.leaves():
                    str2 += (word[0] + ' ')
                
                pair_noun_phrases.append((str1,str2))
                data.append(features)

        doc_num+=1

    return data, pair_noun_phrases