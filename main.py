# coded by jonathan & carlos C511
import sklearn
import nltk
import os
from nltk.stem import WordNetLemmatizer

# languages
ES = 'spanish'
EN = 'english'
# grammars
simple_en_grammar = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
"""
en_grammar = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  PP: {<IN><NP>}               # Chunk prepositions followed by NP
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
  CLAUSE: {<NP><VP>}           # Chunk NP, VP
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

# ! Task 0: load corpus...
def load_corpus():
    for filename in os.listdir(directory+"\\corpus"):
        with open(directory+"\\corpus\\"+filename, "r", encoding='utf8', errors='ignore') as f:
            text = f.read()
            corpus.append(text)

# ! Task 1: Tokenization and sentences segmentation
def tokenization_and_sentence_segmentation():
    for i in range(len(corpus)):
        sentences_corpus.append(nltk.sent_tokenize(corpus[i],language=EN))        
        tokenize_sentence = []
        for j in range(len(sentences_corpus[i])):
            tokenize_sentence.append(nltk.word_tokenize(sentences_corpus[i][j],language=EN))
        token_corpus.append(tokenize_sentence)

# ! Task 2: Morphological Processing
def morphological_processing():
    for i in range(len(token_corpus)):
        sentence = []
        for j in range(len(token_corpus[i])):
            word = []
            for k in range(len(token_corpus[i][j])):
                word.append(wnl.lemmatize(token_corpus[i][j][k]))
            sentence.append(word)
        morpho_token_corpus.append(sentence)        

# ! Task 3: Postagger
def post_tagger():
    for i in range(len(morpho_token_corpus)):
        sentence = []     
        for j in range(len(morpho_token_corpus[i])):
            sentence.append(nltk.pos_tag(morpho_token_corpus[i][j],lang='eng'))
        pos_tag_token.append(sentence)

# ! Task 4: Noun Phrase Identification
def noun_phrase_identification():
    cp = nltk.RegexpParser(en_grammar)
    for i in range(len(pos_tag_token)):
        sentences_NP = []
        for j in range(len(pos_tag_token[i])):
            sentence = []
            tree = cp.parse(pos_tag_token[i][j])
            for subtrees in tree.subtrees():
                if subtrees.label() == 'NP':
                    sentence.append(subtrees)
            sentences_NP.append(sentence)
        noun_phrase_sentences.append(sentences_NP)

# ! Task 5: Named Entity Recognition
def named_entity_recognition():
    # usar nltk.ne_chunk(tagged_sentence)
    pass

# ! Task 6: Nested Noun Phrase Extraction
def nested_noun_phrase_extraction():
    pass

# ! Task 7: Semantic Class Determination
def semantic_class_determination():
    pass

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
    nested_noun_phrase_extraction()
    semantic_class_determination()
    get_markables()