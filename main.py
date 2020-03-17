#coding:utf-8
#-*- coding : utf-8 -*-
#朴素贝叶斯算法
import pandas as pd

class NaiveBayes(object):
    def getData(self,filename):
        dataSet = pd.read_csv(filename,encoding="GBK")
        dataSetDF = pd.DataFrame(dataSet)
        dataSetDF.drop(['id'],axis=1,inplace=True)  #删除id
        dataSetDF.drop(['RBP4'],axis=1,inplace=True)

        #datalabel = dataSetDF["label"]
        #dataSetDF.drop(['label'], axis=1, inplace=True)#删除label
        #将不连续数据离散化
        #dataSetDF["RBP4"] = pd.cut(dataSetDF["RBP4"], 10,labels=[1,2,3,4,5,6,7,8,9,10])

        dataSetDF["年龄"] = pd.cut(dataSetDF["年龄"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["身高"] = pd.cut(dataSetDF["身高"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["孕前体重"] = pd.cut(dataSetDF["孕前体重"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["孕前BMI"] = pd.cut(dataSetDF["孕前BMI"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["收缩压"] = pd.cut(dataSetDF["收缩压"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["舒张压"] = pd.cut(dataSetDF["舒张压"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["分娩时"] = pd.cut(dataSetDF["分娩时"], 5,labels=[1,2,3,4,5])
        dataSetDF["糖筛孕周"] = pd.cut(dataSetDF["糖筛孕周"], 5,labels=[1,2,3,4,5])
        dataSetDF["VAR00007"] = pd.cut(dataSetDF["VAR00007"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["wbc"] = pd.cut(dataSetDF["wbc"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["ALT"] = pd.cut(dataSetDF["ALT"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["AST"] = pd.cut(dataSetDF["AST"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["Cr"] = pd.cut(dataSetDF["Cr"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["BUN"] = pd.cut(dataSetDF["BUN"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["CHO"] = pd.cut(dataSetDF["CHO"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["TG"] = pd.cut(dataSetDF["TG"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["HDLC"] = pd.cut(dataSetDF["HDLC"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["LDLC"] = pd.cut(dataSetDF["LDLC"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["ApoA1"] = pd.cut(dataSetDF["ApoA1"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["ApoB"] = pd.cut(dataSetDF["ApoB"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["Lpa"] = pd.cut(dataSetDF["Lpa"], 10,labels=[1,2,3,4,5,6,7,8,9,10])
        dataSetDF["hsCRP"] = pd.cut(dataSetDF["hsCRP"], 10,labels=[1,2,3,4,5,6,7,8,9,10])


        return dataSetDF




    def classify(self,trainData,testData):
        #朴素贝叶斯公式P(C|X)=(P(C)/P(X))*∏_(i=1)^d▒〖P(x_i 〗|c)
    #进行训练
        #计算P(Ci)概率
        Probability_C = {}
        lables = list(trainData['label'])
        for label in range(2):
            Probability_C[label] = lables.count(label)/float(len(lables))
        #按label划分数据集
        trainData_C0 = trainData[trainData['label'].isin([0])]
        rowsize_C0 = trainData_C0.iloc[:,0].size
        trainData_C1 = trainData[trainData['label'].isin([1])]
        rowsize_C1 = trainData_C1.iloc[:,0].size

        #计算P(x_i|Ci)的概率
        Probability_xi_C0 = {}
        Probability_xi_C1 = {}

        features = trainData.columns.values#存放列表名
        #计算C0类别下每个特征的概率
        for feature in features:
            sta_xi = trainData_C0[feature].value_counts()
            sta_xi_index = sta_xi.index
            Probability_xi_C0[feature] = sta_xi[sta_xi_index] / rowsize_C0 #?????

        #计算C1类别下每个特征的概率
        for feature in features:
            sta_xi = trainData_C1[feature].value_counts()
            sta_xi_index = sta_xi.index
            Probability_xi_C1[feature] = sta_xi[sta_xi_index]/rowsize_C1


    #进行分类
        features = testData.columns.values#
        result = []
        for row in testData.itertuples():
            #计算类别概率
            p0 = Probability_C[0]
            p1 = Probability_C[1]
            t0=1
            t1=1
            for feature in features:
                if(Probability_xi_C0[feature][getattr(row,feature)]==0):
                    t0 = t0 /rowsize_C0
                else:
                    t0 = t0 * Probability_xi_C0[feature][getattr(row, feature)]
                if(Probability_xi_C1[feature][getattr(row,feature)]==0):
                    t1 = t1 /rowsize_C1
                else:
                    t1 = t1 * Probability_xi_C1[feature][getattr(row, feature)]

            p0 = t0*p0
            p1=  t1*p1

            if (p0 >= p1):
                result.append(0)
            else :
                result.append(1)
        return  result




if __name__ == '__main__':
    nb = NaiveBayes()
    # 训练数据
    trainData = nb.getData('f_train.csv')
    label_right = pd.read_csv('f_testlabel.csv',encoding="GBK")
    label_right = pd.DataFrame(label_right)
    label_right = list(label_right['label'])
    testData = nb.getData('f_test.csv')
    result = nb.classify(trainData,testData)
    print('The number of train sample is:',len(trainData['label']))
    print('The result is:',result)
    print('The number of test sample is:',len(result))
    count = 0
    for i in range(len(label_right)):
        if( result[i]==label_right[i]):
            count = count+1
    accuracy = count/len(result)
    print('Accuracy is :',accuracy)
    result = pd.DataFrame(result)
    result.to_csv('f_answer.csv',index=False,header=None)
