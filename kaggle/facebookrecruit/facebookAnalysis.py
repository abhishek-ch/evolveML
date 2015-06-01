import graphlab
import threading
__author__ = 'abhishekchoudhary'
graphlab.canvas.set_target('browser')

sf = graphlab.SFrame.read_csv('/Users/abhishekchoudhary/Work/python/facebook/trainingset.csv', header=True)
sf.show()
