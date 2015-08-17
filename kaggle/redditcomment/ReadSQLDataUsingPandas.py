__author__ = 'achoudhary'
<<<<<<< HEAD
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import pylab

fig = plt.figure()

db = '../input/database.sqlite'
conn = sqlite3.connect(db)
cur = conn.cursor()

df = pd.read_sql(('SELECT * FROM May2015 \
WHERE rowid % 6000 = 0 and score<1000 \
ORDER BY created_utc'),conn)

print(df.axes[1])

plt.plot(df.created_utc, df.score)

fig.savefig('graph.png')
=======
import re as jet
import sqlite3 as fuel
import matplotlib.pyplot as cant
import numpy as melt
from collections import Counter as steel


dank = fuel.connect('../input/database.sqlite')

meme = "SELECT lower(body)      \
    FROM May2015                \
    WHERE LENGTH(body) < 40     \
    and LENGTH(body) > 20       \
    and lower(body) LIKE 'jet fuel can''t melt%' \
    LIMIT 100";

beams = []
for illuminati in dank.execute(meme):
    illuminati = jet.sub('[\"\'\\,!\.]', '', (''.join(illuminati)))
    illuminati = (illuminati.split("cant melt"))[1]
    illuminati = illuminati.strip()
    beams.append(illuminati)

bush = steel(beams).most_common()
labels, values = zip(*bush)
indexes = melt.arange(len(labels))

cant.barh(indexes, values)
cant.yticks(indexes, labels)
cant.tight_layout()
cant.savefig('dankmemes1.png')
>>>>>>> 78d6e223043bf30c18643cebbdf4661f501aadd7
