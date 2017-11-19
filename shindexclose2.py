# -*- coding: utf-8  -*-

import os
import numpy as np
import random
import tensorflow as tf
import sys
allDataSet=[]

n_input = 1    # input data vector is 1, [close]
lstm_step = 30   # 30 days data as time seq
n_hidden = 128   # LSTM  hidden cells
n_output = 21  #  output is a 41d vector, fron 10% to -10%, step is 1.0%
               #  the value can be 11 21 41 , caculator step automatic
               #  if it is 3 ,  step should be -0.5 +0.5
batch_size = 64

testDataAccount = 40  # how many data account of one stock ,for forming a testData set

allSamples=[]  # all samples
allLabels=[]   # all labels ,for every code , less 1 of length of code day data
accountSamples=0
accountArray=[]

toVarray=np.array([],dtype='float32')

cur_ind = 0
# data as [[[stock_code1,date,open,close,high,low,turnover,vol,amount],
#           [stock_code1,                                            ],
#           [ ....                                                   ]],
#          [[stock_code2,date,open,close,high,low,turnover,vol,amount],
#           [ ....                                                   ]],
#
#
#           [ ...                                                    ]]]
def loadData(datapath):
    flist = os.listdir(datapath)
    datas=[]
    for afile in flist:
        f = open(datapath+"/"+afile)
        alllines = f.readlines()
        f.close()
        del alllines[0]
        onestocklist=[]
        firstrec = 1
        delcontinue10 = 0
        for oneline in alllines:

            oneline = oneline.rstrip('\n')
            aline = oneline.split(',')
            onestocklist.append(aline)
        
        datas.append(onestocklist)
    return datas

#convert all data list to np.array data
def convert2array(datas):
    allarray = []              # this is a python list,because every stock code has different trading days, length is diffrent
    for onecode in datas:
        onecodearray=np.array([[]],dtype='float32')
        for oneday in onecode:
            onedaycpy = oneday[4:5]
#            print(onedaycpy)
            onedayarray = np.array(onedaycpy,dtype='float32')
            if len(onecodearray[0]) == 0:
                onecodearray = np.array([onedayarray])
            else:
                onecodearray = np.append(onecodearray,[onedayarray],axis=0)
#        print(onecodearray)
        allarray.append(onecodearray)
#   allarray is a python list,and every element is a np.array, that is one stock code array  as
#                           [[open,close,high,low,turnover],....,[open,close,high,low,turnover]]

    alllabels = []
    for code in allarray:
        onecodelabels = np.array([[]],dtype='float32')
        lastdayclose = code[0][0]      #get the first close price of a stock-code
        i = 0
        for everyday in code:
            if i==0:
                i=1
                continue
            ratio = (everyday[0]-lastdayclose)/lastdayclose
            lastdayclose = everyday[0]
#            print(ratio)
            ratio = calonehot(ratio)       # change to a one hot  vector
#            print(ratio)
            if len(onecodelabels[0]) == 0 :
                onecodelabels = np.array([ratio],dtype='float32')
            else:
                onecodelabels = np.append(onecodelabels,[ratio],axis=0)
          
        alllabels.append(onecodelabels)
    for code in allarray:
        baseprice = code[0][0]
        for everyday in code:
            everyday[0]=everyday[0]/baseprice
#    print "data len"
#    print(len(allarray[0]))
#    print(len(alllabels[0]))
    return allarray,alllabels

# according dimention of output,cacluate the step point of up to down tick
def productVarray(outputd):
    _toVarray=np.array([],dtype='float32')
    if outputd == 3:
        _toVarray = np.append(_toVarray,[-0.3])
        _toVarray = np.append(_toVarray,[0.3])
        return _toVarray
    fstep = 20.0/(outputd-1)
    fstart=-10.0 + fstep/2
    for i in range(outputd-1):
        _toVarray = np.append(_toVarray,[fstart])
        fstart = fstart + fstep
    return _toVarray

def calonehot(upordown):
    i = 0
    change = 0
    retval = np.zeros([n_output],dtype='float32')
    upordown = upordown * 100
    for f in toVarray:
        if (upordown < f):
            retval[i]=1.0
            return retval
        i = i + 1
    retval[i] = 1.0
    return retval

# onehot to ratio value
def onehot2ratio(indexofonehot):
    ret  = 0
    if indexofonehot == 0:
        ret = (-10.0+toVarray[0])/2
    elif indexofonehot == n_output-1:
        ret = (10.0+toVarray[n_output-1])/2
    else:
        ret = (toVarray[indexofonehot-1]+toVarray[indexofonehot])/2
    return ret

def toNormalize(alldatas):
    for onecode in alldatas:
        _maxprice = 0
        for everyday in onecode:
            if _maxprice <= everyday[0]:
                _maxprice = everyday[0]
        for everyday in onecode:
            everyday[0] = everyday[0] / _maxprice

def toNormalize2(alldatas):   # methord:  initday is zero
    for onecode in alldatas:
        _firstday = onecode[0][0]
        for everyday in onecode:
            everyday[0] = everyday[0] / _firstday - 1.0
            everyday[1] = everyday[1] / _firstday - 1.0
            everyday[2] = everyday[2] / _firstday - 1.0 
            everyday[3] = everyday[3] / _firstday - 1.0

# give a up or down percent, change to a onehot vector
#def calonehot(upordown):
#    r = upordown * 100
#    r = r + 10
#    if r<0:
#        r = 0
#    r = r * 2
#    i = int(r+0.5)
#    if (i>40):
#        i = 40
#    retval = np.zeros([n_output],dtype='float32')
#    retval[i] = 1
#    return retval


# in every code, get last 2 sample for test ,   -32 to -3,  -31 to -2  
#  samples size is code num * lstm_step * 2,  lables size is n_output * 2
def prepareTestSet(alldatas,alllabels):
    TestSamples=np.array([],dtype='float32')
    TestLabels=np.array([],dtype='float32')
    for onecodearray in alldatas:
        i = testDataAccount
        for j in range(testDataAccount):
            getdata = onecodearray[-lstm_step-i:-i]
            TestSamples = np.append(TestSamples,getdata)
            i = i - 1
#        getdata = onecodearray[-lstm_step-2:-2]
#        TestSamples = np.append(TestSamples,getdata)
#        getdata = onecodearray[-lstm_step-1:-1]
#        TestSamples = np.append(TestSamples,getdata)
    for onecodelabels in alllabels:
        getdata = onecodelabels[-testDataAccount:]
        TestLabels = np.append(TestLabels,getdata)
    return np.reshape(TestSamples,[-1,lstm_step,n_input]),np.reshape(TestLabels,[-1,n_output])

# Total Train count is:  length - lstem_step -2
# the first group index is [from 0 to lstem_step-1] and label index is lstem_step -1 
def calTotal(alldatas,alllabels):
    accountSamples = 0
    accountarray = np.array([],dtype=int)
    for onecodearray in alldatas:
        accountSamples = accountSamples+len(onecodearray)-lstm_step-testDataAccount
        accountarray = np.append(accountarray,[accountSamples])
    return accountSamples,accountarray

#form one seque for predict
def form1seq(seqdata):
    seqset=np.array([],dtype='float32')
    seqset=np.append(seqset,seqdata)
    return np.reshape(seqset,[-1,lstm_step,n_input]) 

#  fetch a seq data and labels, length is lstm_step
def fetchSeq(alldatas,alllabels,account,num):
    i = num
    j = 0
    k = num
    lastl = 0
    for l in account:
        if i<l:
            k = k - lastl
            break
        j = j + 1
        lastl = l
    if j >= len(account):
        return
    datagroup = np.array([],dtype="float32")
    for i in range(k,k+lstm_step):    # fetch from k ,to k+lstem_step-1
        datagroup = np.append(datagroup,alldatas[j][i])
    labelgroup = alllabels[j][k+lstm_step -1]
    return np.reshape(datagroup,[-1,lstm_step,n_input]),np.reshape(labelgroup,[-1,n_output]),j

#getbatch seq
def getbatch(batchsize):
    global cur_ind
    batchSam = np.array([],dtype='float32')
    batchLab = np.array([],dtype='float32')
    for i in range(batchsize):
        Sam,Lab,_stockind = fetchSeq(allSamples, allLabels, accountArray,cur_ind)
        cur_ind = cur_ind +1
        if (cur_ind >= accountSamples):
            cur_ind = 0 
        batchSam = np.append(batchSam,Sam)
        batchLab = np.append(batchLab,Lab)
    return np.reshape(batchSam,[-1,lstm_step,n_input]),np.reshape(batchLab,[-1,n_output])

#getbatch random
def getbatchrandom(batchsize):
    batchSam = np.array([],dtype='float32')
    batchLab = np.array([],dtype='float32')
    for i in range(batchsize):
        ind = random.randint(0,accountSamples-1)
        Sam,Lab,_stockind = fetchSeq(allSamples, allLabels, accountArray,ind)
        batchSam = np.append(batchSam,Sam)
        batchLab = np.append(batchLab,Lab)
    return np.reshape(batchSam,[-1,lstm_step,n_input]),np.reshape(batchLab,[-1,n_output])

x = tf.placeholder("float",[None,lstm_step,n_input])
y = tf.placeholder("float",[None, n_output])
istate = tf.placeholder("float", [None, 2 * n_hidden])

weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden,n_output]))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

def RNN(_X, _istate, _weight, _biases):
    _xt = tf.transpose(_X,[1,0,2])
    _xr = tf.reshape(_xt, [-1,n_input])
    _xm = tf.matmul(_xr, _weight['hidden'])+_biases['hidden']
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell,lstm_cell2])    
    _x = tf.split(_xm,lstm_step,0)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell,_x,dtype=tf.float32)
    return tf.matmul(outputs[-1],_weight['out'])+_biases['out']

toVarray=productVarray(n_output)
dataFilePath="./index/sh"
#dataFilePath="./bigbank"
#dataFilePath="s:/pricedata/financestock/d_pre"
modelpath="models/shindex.ckpt"

allDataSet = loadData(dataFilePath)
print(allDataSet[0][0])
print(allDataSet[0][1])
allSamples,allLabels = convert2array(allDataSet)
print(allSamples[0][0])
toNormalize(allSamples)
print(allSamples[0][0])
accountSamples,accountArray = calTotal(allSamples,allLabels)
print(accountSamples)   # total learn datas  27844
print(accountArray)
TestV,TestL = prepareTestSet(allSamples,allLabels)  # this is the test data , code number * 2
#dd,ll=fetchSeq(allSamples,allLabels,accountArray,0)
#dd,ll=getbatch(32)
#print(dd)
#print(ll)


#print(len(TestV))
#print(TestV[1])
#print(len(TestL))
#print(TestL[0])

pred = RNN(x,istate,weights,biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.RMSPropOptimizer(0.0001,0.9).minimize(cost)
#optimizer = tf.train.AdamOptimizer(0.0001,0.9).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

prediction = tf.argmax(pred,1)

if len(sys.argv) == 1:



    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    # Launch the graph in a session
    with tf.Session(config=session_conf) as sess:

    #with tf.Session() as sess:
        sess.run(init)
        step = 1
        while step <= 150000:
            step = step + 1
            batch_xs, batch_ys = getbatch(batch_size)
            sess.run(optimizer,feed_dict={x:batch_xs, y:batch_ys,istate:np.zeros((batch_size,2*n_hidden))})
            if step % 100 == 0:
                acc = sess.run(accuracy,feed_dict={x:batch_xs, y:batch_ys,istate:np.zeros((batch_size,2*n_hidden))})
                loss = sess.run(cost, feed_dict={x:batch_xs, y:batch_ys,istate:np.zeros((batch_size,2*n_hidden))})
                print("Iter: "+str(step)+" Loss= "+"{:0.5f}".format(loss)+"  Accuracy="+"{:0.5f}".format(acc))
        print("Optimization Finished!")
        print("Saving model...")
        saver_path = saver.save(sess,modelpath)
        print("Saved in file:",saver_path)
        print("开始预测...")
        print("Test Label is :")
        print(TestL)
        accu = sess.run(accuracy,feed_dict={x:TestV, y:TestL,istate:np.zeros((batch_size,2*n_hidden))})
        thispred = sess.run(pred,feed_dict={x:TestV, y:TestL,istate:np.zeros((batch_size,2*n_hidden))})
        print("predict:")
        print(thispred)
        print("Accuracy is "+"{:0.5f}".format(accu))
elif sys.argv[1] == 'learn':
    print('learn')
elif sys.argv[1] == 'predict':
    print('predict')
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    with tf.Session(config=session_conf) as sess:
        saver.restore(sess,modelpath)
        print("开始预测...")
        print("Test Label is :")
        print(TestL)
        accu = sess.run(accuracy,feed_dict={x:TestV, y:TestL,istate:np.zeros((batch_size,2*n_hidden))})
        thispred = sess.run(prediction,feed_dict={x:TestV,istate:np.zeros((batch_size,2*n_hidden))})
        print("predict:")
        print(thispred)
#        for a in thispred:
#            print(onehot2ratio(a))
        print("Accuracy is "+"{:0.5f}".format(accu))
elif sys.argv[1] == 'showdata':

    # this is for show data by matplotlib
    import matplotlib.pyplot as plt
    xais = 220
    pltx = np.zeros([xais])
    for i in range(xais):
        pltx[i]=i
    origDataLen=220
    ind = len(allSamples[0])-origDataLen
    pltdata=np.array([],dtype='float32')
    d = allSamples[0]
    for i in range(origDataLen):
        pltdata=np.append(pltdata,[d[ind+i][0]])
    plt.plot(pltx,pltdata,'r')

    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    with tf.Session(config=session_conf) as sess:
        saver.restore(sess,modelpath)
        print("开始预测...")
        print("Test Label is :")
        print(TestL)
        accu = sess.run(accuracy,feed_dict={x:TestV, y:TestL,istate:np.zeros((batch_size,2*n_hidden))})
        thispred = sess.run(prediction,feed_dict={x:TestV,istate:np.zeros((batch_size,2*n_hidden))})
        print("predict:")
        print(thispred)
        
#        for a in thispred:
#            print(onehot2ratio(a))
    testdata = np.array([],dtype='float32')
    testx = np.zeros([testDataAccount])
    d=allSamples[0]
    j = len(d)-testDataAccount  # the last price index 
    for i in range(testDataAccount):
        pindex = d[j][0]*(1+onehot2ratio(thispred[i])/100.0)
        j=j+1
        testdata=np.append(testdata,[pindex])
        testx[i]=origDataLen-testDataAccount+i
    plt.plot(testx,testdata,'k')

    with tf.Session(config=session_conf) as sess:
        saver.restore(sess,modelpath)
        print("开始预测...")
        predictdata=np.array([],dtype='float32')
        for j in range(testDataAccount):
            seq=d[-lstm_step-testDataAccount:-testDataAccount]
            print(seq)
            seqdata=np.array([seq],dtype='float32')
            seqSet=form1seq(seqdata) 
            thispred = sess.run(prediction,feed_dict={x:seqSet,istate:np.zeros((batch_size,2*n_hidden))})
            print("predict:")
            print(thispred)
            newprice = seq[lstm_step-1][0] * (1+onehot2ratio(thispred[0])/100.0)
            predictdata=np.append(predictdata,[newprice])
            for i in range(lstm_step-1):
                seq[i][0]=seq[i+1][0]
            seq[lstm_step-1] = newprice
        print(predictdata)
    plt.plot(testx,predictdata,'b')


#print(Sam)
#print(Sam)
#print(accountArray)
    plt.grid()
    plt.show()
else:
    print('Argument error!')

