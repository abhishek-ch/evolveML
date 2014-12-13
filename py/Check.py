__author__ = 'abhishekchoudhary'
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import pandas as pd



input = [u'x',
         u"RT @jvsk3: Banning is d way 2 hide once incompetency's &amp; BJP ban is 2 hide corruption which led 2 crime  https://t.co/VOR9EVll9U #DelhiSham ",
         u'RT @NoName4840:                  (      ) https://t.co/WrCrzRrhpM #IslamicState #IS #ISIS #ISIL #ISID #India  #DelhiShamedAgain',
         u'RT @DelhiDialogue: BJP-led SDMC. Another poor municipal corporation. Women bear the worst of their ineffectiveness.. #DelhiShamedAgain http ',
         u'RT @DelhiDialogue: CCTVs in buses and bus shelters. Make private taxis redundant by making public transport safe. #DelhiShamedAgain http:// ',
         u'"RT @DrunkVinodMehta: Cab Driver was acquitted by Court in earlier rape case. If Modi be PM after 2002 riots accuse, why can\'t he drive a Ca "',
         u'"RT @DrunkVinodMehta: NihalChand accused of Rape, eligible for Cabinet Minister. Cab Driver accused of rape, #Uber shld not hv employed him. "',
         u'RT @jvsk3: #DelhiShamedAgain If character certificate was not issued by Delhi police would accused cab driver got d job?  https://t.co/VOR9 ',
         u'Passport address verification is done by giving 500  to police. Do u think they will do driver verification diligently ? #DelhiShamedAgain',
         u"RT @DelhiDialogue: BJP-led NDMC exhibits poor fiscal prudence on women's issues when it needs proactive governance. #DelhiShamedAgain http: "]


df = pd.DataFrame(input[1:])
print df.head()