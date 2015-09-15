__author__ = 'achoudhary'

def check():
    #global time
    time.append(1)
    time.append(2)

def printchck():
    #global time
    for val in time:
        print val

if __name__ == "__main__":
    time = []
    check()
    printchck()