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
  NP: {(<PRP>)|(<DT>?<JJ>*<NN.*>+<POS><JJ>*<NN.*>+)|(<DT|PRP\$>?<JJ>*<NN.*>+)}     # chunk determiner/possessive, adjectives and noun
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

# ! Task 0: load corpus... ok!
def load_corpus():
    for filename in os.listdir(directory+"\\corpus"):
        with open(directory+"\\corpus\\"+filename, "r", encoding='utf8', errors='ignore') as f:
            text = f.read()
            corpus.append(text)

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
                word.append(wnl.lemmatize(token_corpus[i][j][k]))
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

# ! Task 6: Nested Noun Phrase Extraction
def find_pos_pron(tree):
    l = []
    for i in range(len(tree)):
        if tree[i][1] == 'PRP$':
            n_tree = nltk.Tree('NP',tree[i])
            l.append(n_tree)
            continue
        if tree[i][1] == 'POS' and i>0:
            n_tree = nltk.Tree('NP',tree[i-1])
            l.append(n_tree)
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
    # semantic_class_determination()
    # get_markables()
    print(str(corpus))
    print(str(token_corpus))
    print(str(morpho_token_corpus))
    print(str(pos_tag_token))
    print(str(noun_phrase_sentences))
    print(str(entities_in_sentence))

main()