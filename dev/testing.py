import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn


# wnl = WordNetLemmatizer()
# word = []
# word.append(wnl.lemmatize('dogs'))
# print(str(word))



# sentence = 'this is the house, that man, these village, those people'
# sentence = 'I, you, he, she, it, we, they, me, him, her, us, them' # personal
# sentence = 'me, him, her, it, us, you, them, whom, mine, yours, his, hers, ours, theirs'
# sentence = 'this is the house, that man, these village, those people'
# sentence = 'this is the house, that man, these village, those people'
# print(sentence)
# tokens = nltk.word_tokenize(sentence,language='english')
# tokens_tagged = nltk.pos_tag(tokens,lang='eng')
# print(str(tokens))
# print(str(tokens_tagged))

# oracion = 'El pequeño perro amarillo le ladró al gato'
# print(oracion)
# palabras = nltk.word_tokenize(oracion,language='spanish')
# palabras_catalogadas = nltk.pos_tag(palabras,lang='spa')
# print(str(palabras))
# print(str(palabras_catalogadas))

# grammar = r"""
#   NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
#   """
# cp = nltk.RegexpParser(grammar,loop=2)
# sentence = [("Mary", "NN"),("Elizabeth", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),
#     ("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]

# print(cp.parse(sentence))

# sent = nltk.corpus.treebank.tagged_sents()[22]
# print(sent)

# for id in nltk.corpus.names.fileids():
#         with open(id,"r") as f:
#             lines = (f.readlines()).split('\n')
#             print(lines)

# male = wn.synset('male.n.02')
# female = wn.synset('female.n.02')
# location = wn.synset('location.n.01')
# organization = wn.synset('organization.n.01')
# he = wn.synset('he..01')

# print(male.definition())
# print(female.definition())
# print(location.definition())
# print(organization.definition())
# print(he.definition())

mr = wn.synset('Mr.n.01')
print(mr.definition())