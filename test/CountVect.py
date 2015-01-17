__author__ = 'abhishekchoudhary'
import re
from sklearn.feature_extraction.text import CountVectorizer

tags = [
  "python tools",
  "linux, tools, ubuntu",
  "distributed systems, linux, networking, tools",
]


REGEX = re.compile(r",\s*")
def tokenize(text):
    print('ayya ',text)
    return [tok.strip().lower() for tok in REGEX.split(text)]

vec = CountVectorizer(tokenizer=tokenize)
data = vec.fit_transform(tags).toarray()
print(data)