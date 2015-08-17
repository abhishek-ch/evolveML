__author__ = 'achoudhary'
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
