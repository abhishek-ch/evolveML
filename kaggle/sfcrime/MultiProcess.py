__author__ = 'abc'
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
import shapefile
import xml.etree.ElementTree as etree
import re
import pandas as pd

filename=""
i_=0

#housing_a = pd.read_csv("../input/pums/ss13husa.csv").fillna(" ")
#housing_b = pd.read_csv("../input/pums/ss13husb.csv").fillna(" ")
#population_a = pd.read_csv("../input/pums/ss13pusa.csv").fillna(" ")
#population_b = pd.read_csv("../input/pums/ss13pusb.csv").fillna(" ")

def data_files():
    t= etree.parse(filename+".shp.xml")
    r = t.getroot()
    st_xml = r.find("./idinfo/descript/Subject_Entity")
    n = re.sub(" |\(|\)","_", st_xml.text)
    print(n)
    r = shapefile.Reader(filename)
    #shapes = r.shapes()
    #print(len(shapes))
    #fields = r.fields
    #print(fields)

    fig = plt.figure()
    #fig.set_size_inches(8, 8)
    for shape in r.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x,y)
    #plt.savefig('plot'+str(i_)+'.png')
    plt.savefig(n+'.png')
    return

if __name__ == '__main__':
    j = []
    states=['01','02','04','05','06','08','09','10','11','12','13','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','44','45','46','47','48','49','50','51','53','54','55','56','66','72','78']
    cpu = multiprocessing.cpu_count()
    #print (cpu)
    for s_ in range(0,len(states),cpu):
        for i in range(cpu):
            i_=s_+i
            if (i_)<len(states):
                p = multiprocessing.Process(target=data_files)
                filename="../input/shapefiles/pums/tl_2013_"+states[s_+i]+"_puma10"
                j.append(p)
                p.start()
