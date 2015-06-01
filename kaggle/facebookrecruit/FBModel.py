__author__ = 'abhishekchoudhary'

import os
import pandas as pd
import graphlab as gl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from MultiColumnLabelEncoder import MultiColumnLabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier,GradientBoostingClassifier)
from sklearn.feature_extraction import DictVectorizer


#gl.canvas.set_target('browser')
if __name__ == '__main__':
    BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..', 'facebook'))

    gl_frame = gl.SFrame.read_csv(BASE_DATA_PATH+"/bids.csv", header=True)
    #bidsdf = gl_frame.to_dataframe() #convert to dataframe
    gl_train = gl.SFrame.read_csv(BASE_DATA_PATH+"/trainingset_clean.csv", header=True)
    train = pd.read_csv(BASE_DATA_PATH + "/trainingset_clean.csv") #load feature extracted dataset
    test = pd.read_csv(BASE_DATA_PATH + "/test.csv")  #load the test dataset

    #df.iloc[:,[2,3,4]]
    X = train.iloc[:,[3,4,5,6,7,8,9]]
    y = train.iloc[:,[9]]
    # flatten y into a 1-D array
    y = np.ravel(y)

    '''
    X = X.T.to_dict().values()
    # turn list of dicts into a numpy array
    vect = DictVectorizer(sparse=False)
    X = vect.fit_transform(X)
    '''

    dataFr = pd.DataFrame(columns=('bidder_id','prediction'))
    model = gl.logistic_classifier.create(gl_train, target='outcome', features=['bidder_id', 'auction', 'merchandise','device','time','country','ip','url'],validation_set=None)

    #values = test.bidder_id[1:4]
    #count = 0
    val = ['b24e3af20453813e821f5a22ff55e072tgton']
    index = 0
    for index,row in test.iterrows():
        bidderrow = gl_frame.filter_by(row.bidder_id,'bidder_id')
        if len(bidderrow) == 0:
            print 'no details {}'.format(row.bidder_id)
            output = '0.0'
        else:
            predicitons = model.predict(bidderrow,output_type='probability')
            output = "{0:.1f}".format(predicitons.astype(float).mean())
            print '=>{} {} len {} mean {} index {}'.format(row.bidder_id,len(predicitons),predicitons.mean(),output,index+1)
        dataFr.loc[index+1] = [row.bidder_id,output]

    print dataFr[1:10]
    dataFr.to_csv(BASE_DATA_PATH+'/Submission_LinearReg.csv', sep=',', encoding='utf-8')

    '''
    bidderrow = gl_frame.filter_by(test.bidder_id[1:4],'bidder_id')

    predicitons = model.predict(bidderrow,output_type='probability')
    #results = model.evaluate(bidderrow)

    print predicitons
    #df_gl = predicitons.to_dataframe()
    value = "{0:.1f}".format(predicitons.astype(float).mean())
    print '{} {} {}'.format(len(predicitons),predicitons.mean(),value)
    '''
    #bidderrowdf = bidderrow.to_dataframe()
    #print(bidderrowdf[1:4])

    #multi = MultiColumnLabelEncoder(columns = ['auction','merchandise','device','country','ip','url']).fit_transform(bidderrowdf)
    #print multi

    '''
    le = preprocessing.LabelEncoder()
    columns=['auction','merchandise','device','country','ip','url']
    for col in columns:
        bidderrowdf[col] = LabelEncoder().fit_transform(bidderrowdf[col])

    for col in columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    '''

    '''
    pipeline = Pipeline([
        ('encoding',MultiColumnLabelEncoder(columns=['auction','merchandise','device','country','ip','url'])),
        ('classifier', LogisticRegression())
        # add more pipeline steps as needed
    ])
    '''
    #print(bidderrowdf[1:4])

    #working Scikit-learn API use
    '''

    classifier = LogisticRegression()
    print 'Classification Starting...'
    classifier = GradientBoostingClassifier()
    model = classifier.fit(X, y)
    print 'classification done'
    # check the accuracy on the training set
    print model.score(X, y)

    predicted = model.predict(bidderrowdf.iloc[:,[3,4,5,6,7,8,9]])
    print(predicted)

    '''