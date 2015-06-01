__author__ = 'abhishekchoudhary'
#using Graphlab API to read the csv and fetch the required data to proceed.
import graphlab as gl
import threading
import os
import pandas as pd

#graphlab.canvas.set_target('browser')




if __name__ == '__main__':
    BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..', 'facebook'))
    gl_frame = gl.SFrame.read_csv(BASE_DATA_PATH+"/bids.csv", header=True)
    #gl_frame_df = gl_frame.to_dataframe() #convert to dataFrame
    #print(BASE_DATA_PATH)
    train = pd.read_csv(BASE_DATA_PATH + "/train.csv")
    #trainrows = train.iterrows()
    df = []
    count = 0

    #print(train.bidder_id[train.outcome == 1][1:4])

    INPUT = [0,1]
    for val in INPUT:
        print '{}{}'.format("updating => ",val)
        bidderrow = gl_frame.filter_by(train.bidder_id[train.outcome == val],'bidder_id')
        bidrow_df = bidderrow.to_dataframe()
        bidrow_df['outcome'] = pd.Series(val, index=bidrow_df.index)
        df.append(bidrow_df)
        print '{}{}'.format("length = ",len(df))

    df1 = df[0]
    df2 = df[1]
    df3 = df1.append(df2,ignore_index = True)
    print '{}{}{}{}{}{}'.format("BIDROW length ",len(df1),' DF length ',len(df2),' DF3 ',len(df3))
    print "Creating CSV and Updating..."
    df3.to_csv(BASE_DATA_PATH+'/trainingset3.csv', sep=',', encoding='utf-8')
    print("CSV Done !!!")



    #########################################################################################################
    # Quick analysis using GraphLab classification library
    # Getting amazing result of 99.9
    #########################################################################################################

    #Shuffle the data https://dato.com/products/create/docs/generated/graphlab.toolkits.cross_validation.shuffle.html
    data = gl.SFrame('/Users/abhishekchoudhary/Work/python/facebook/trainingset3.csv')
    data = gl.cross_validation.shuffle(data)
    folds = gl.cross_validation.KFold(data, 5)
    for train, valid in folds:
        m = gl.boosted_trees_classifier.create(train, target='label')
        print m.evaluate(valid)

    # Get the i'th fold
    (train, test) = folds[4]
    #do a quick classification analysis on the dataset
    #https://dato.com/products/create/docs/graphlab.toolkits.classifier.html
    model = gl.classifier.create(train, target='outcome', features=['bidder_id', 'auction', 'merchandise','device','time','country','ip','url'])
    #https://dato.com/products/create/docs/generated/graphlab.toolkits.cross_validation.KFold.html#graphlab.toolkits.cross_validation.KFold

    #After above K-fold
    #https://dato.com/products/create/docs/generated/graphlab.boosted_trees_classifier.create.html

    #doing the prediction
    predicitons = model.classify(test)
    results = model.evaluate(test)

    '''
    PROGRESS: Model selection based on validation accuracy:
    PROGRESS: ---------------------------------------------
    PROGRESS: LogisticClassifier              : 0.999522
    PROGRESS: SVMClassifier                   : 0.997674
    PROGRESS: ---------------------------------------------
    '''