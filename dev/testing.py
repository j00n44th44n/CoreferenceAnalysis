import nltk
from nltk.stem import WordNetLemmatizer

# wnl = WordNetLemmatizer()
# word = []
# word.append(wnl.lemmatize('dogs'))
# print(str(word))



# sentence = 'the little yellow dog barked at the cat'
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