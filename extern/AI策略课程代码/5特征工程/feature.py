import datetime
import numpy as np
import pandas as pd


file_name_1 ='m99_1m_TurnoverOI_10946_6928_866.csv'    ###########  轴文件  ##############
df = pd.read_csv(file_name_1)

file_name_2 ='m99_1m_TurnoverOI_10946_6928_866_Label_513.csv'    ############# Label 文件 ###########
df_label = pd.read_csv(file_name_2)

date = df['eob'] > '2023-01-01'    ##########  测试集开始日期，格式：xxxx-xx-xx     #############

date_df = df[date]

Train = len(df) - len(date_df) - 30     #######  减少30个，是为了给特征留出计算的空间
print('Train: ' + str(Train))

Test = len(date_df)
print('Test: ' + str(Test))


file_80 = file_name_1 + '_tz80.csv'    ############ 80个特征过度文件 ############

file_400_Train = file_name_1 + "_tz80_Train_" + str(Train) + ".csv"    ############  训练集文件  #############

file_400_Test = file_name_1 + "_tz80_Test_" + str(Test) + ".csv"    #############  测试集文件 ############

file_400_Train_3 = file_name_1 + "_tz80_Test_" + str(Test) + "_PCA" + ".csv"    #############  测试集针对PCA降维文件 ############

file_Train = str(Train) + ".csv"
file_Test = str(Test) + ".csv"

file_name ='1_400.csv'   
file_temp = pd.read_csv(file_name)
###### 复制第0行N遍 ######
temp_df = file_temp.iloc[0]
for t in range(Train-1): 
        file_temp.loc[file_temp.shape[0]] = temp_df
file_temp.to_csv(str(Train) + ".csv",index = False)



file_name ='1_400.csv'   
file_temp = pd.read_csv(file_name)
###### 复制第0行N遍 ######
temp_df = file_temp.iloc[0]
for t in range(Test-1): 
        file_temp.loc[file_temp.shape[0]] = temp_df
file_temp.to_csv(str(Test) + ".csv",index = False)


#######################################################################################################


df['open'] = df['open'].astype('float64')
df['close'] = df['close'].astype('float64')
df['high'] = df['high'].astype('float64')
df['low'] = df['low'].astype('float64')


open1 = df['open']
close = df['close']
high = df['high']
low = df['low']


eob = df['eob']
datelist = pd.to_datetime(eob[5:])

########################################################################################


logopen1_1 = (np.log(np.array(open1[1:]))-np.log(np.array(open1[:-1])))[4:]
logopen1_2 = (np.log(np.array(open1[1:]))-np.log(np.array(close[:-1])))[4:]
logopen1_3 = (np.log(np.array(open1[1:]))-np.log(np.array(high[:-1])))[4:]
logopen1_4 = (np.log(np.array(open1[1:]))-np.log(np.array(low[:-1])))[4:]

logopen2_1 = np.log(np.array(open1[2:]))-np.log(np.array(open1[:-2]))
logopen2_2 = np.log(np.array(open1[2:]))-np.log(np.array(close[:-2]))
logopen2_3 = np.log(np.array(open1[2:]))-np.log(np.array(high[:-2]))
logopen2_4 = np.log(np.array(open1[2:]))-np.log(np.array(low[:-2]))

logopen3_1 = np.log(np.array(open1[3:]))-np.log(np.array(open1[:-3]))
logopen3_2 = np.log(np.array(open1[3:]))-np.log(np.array(close[:-3]))
logopen3_3 = np.log(np.array(open1[3:]))-np.log(np.array(high[:-3]))
logopen3_4 = np.log(np.array(open1[3:]))-np.log(np.array(low[:-3]))

logopen4_1 = np.log(np.array(open1[4:]))-np.log(np.array(open1[:-4]))
logopen4_2 = np.log(np.array(open1[4:]))-np.log(np.array(close[:-4]))
logopen4_3 = np.log(np.array(open1[4:]))-np.log(np.array(high[:-4]))
logopen4_4 = np.log(np.array(open1[4:]))-np.log(np.array(low[:-4]))

logopen5_1 = np.log(np.array(open1[5:]))-np.log(np.array(open1[:-5]))
logopen5_2 = np.log(np.array(open1[5:]))-np.log(np.array(close[:-5]))
logopen5_3 = np.log(np.array(open1[5:]))-np.log(np.array(high[:-5]))
logopen5_4 = np.log(np.array(open1[5:]))-np.log(np.array(low[:-5]))



logclose1_1 = (np.log(np.array(close[1:]))-np.log(np.array(open1[:-1])))[4:]
logclose1_2 = (np.log(np.array(close[1:]))-np.log(np.array(close[:-1])))[4:]
logclose1_3 = (np.log(np.array(close[1:]))-np.log(np.array(high[:-1])))[4:]
logclose1_4 = (np.log(np.array(close[1:]))-np.log(np.array(low[:-1])))[4:]            

logclose2_1 = np.log(np.array(close[2:]))-np.log(np.array(open1[:-2]))
logclose2_2 = np.log(np.array(close[2:]))-np.log(np.array(close[:-2]))
logclose2_3 = np.log(np.array(close[2:]))-np.log(np.array(high[:-2]))
logclose2_4 = np.log(np.array(close[2:]))-np.log(np.array(low[:-2]))

logclose3_1 = np.log(np.array(close[3:]))-np.log(np.array(open1[:-3]))
logclose3_2 = np.log(np.array(close[3:]))-np.log(np.array(close[:-3]))
logclose3_3 = np.log(np.array(close[3:]))-np.log(np.array(high[:-3]))
logclose3_4 = np.log(np.array(close[3:]))-np.log(np.array(low[:-3]))

logclose4_1 = np.log(np.array(close[4:]))-np.log(np.array(open1[:-4]))
logclose4_2 = np.log(np.array(close[4:]))-np.log(np.array(close[:-4]))
logclose4_3 = np.log(np.array(close[4:]))-np.log(np.array(high[:-4]))
logclose4_4 = np.log(np.array(close[4:]))-np.log(np.array(low[:-4]))

logclose5_1 = np.log(np.array(close[5:]))-np.log(np.array(open1[:-5]))
logclose5_2 = np.log(np.array(close[5:]))-np.log(np.array(close[:-5]))
logclose5_3 = np.log(np.array(close[5:]))-np.log(np.array(high[:-5]))
logclose5_4 = np.log(np.array(close[5:]))-np.log(np.array(low[:-5]))



loghigh1_1 = (np.log(np.array(high[1:]))-np.log(np.array(open1[:-1])))[4:]
loghigh1_2 = (np.log(np.array(high[1:]))-np.log(np.array(close[:-1])))[4:]
loghigh1_3 = (np.log(np.array(high[1:]))-np.log(np.array(high[:-1])))[4:]
loghigh1_4 = (np.log(np.array(high[1:]))-np.log(np.array(low[:-1])))[4:]

loghigh2_1 = np.log(np.array(high[2:]))-np.log(np.array(open1[:-2]))
loghigh2_2 = np.log(np.array(high[2:]))-np.log(np.array(close[:-2]))
loghigh2_3 = np.log(np.array(high[2:]))-np.log(np.array(high[:-2]))
loghigh2_4 = np.log(np.array(high[2:]))-np.log(np.array(low[:-2]))

loghigh3_1 = np.log(np.array(high[3:]))-np.log(np.array(open1[:-3]))
loghigh3_2 = np.log(np.array(high[3:]))-np.log(np.array(close[:-3]))
loghigh3_3 = np.log(np.array(high[3:]))-np.log(np.array(high[:-3]))
loghigh3_4 = np.log(np.array(high[3:]))-np.log(np.array(low[:-3]))

loghigh4_1 = np.log(np.array(high[4:]))-np.log(np.array(open1[:-4]))
loghigh4_2 = np.log(np.array(high[4:]))-np.log(np.array(close[:-4]))
loghigh4_3 = np.log(np.array(high[4:]))-np.log(np.array(high[:-4]))
loghigh4_4 = np.log(np.array(high[4:]))-np.log(np.array(low[:-4]))

loghigh5_1 = np.log(np.array(high[5:]))-np.log(np.array(open1[:-5]))
loghigh5_2 = np.log(np.array(high[5:]))-np.log(np.array(close[:-5]))
loghigh5_3 = np.log(np.array(high[5:]))-np.log(np.array(high[:-5]))
loghigh5_4 = np.log(np.array(high[5:]))-np.log(np.array(low[:-5]))



loglow1_1 = (np.log(np.array(low[1:]))-np.log(np.array(open1[:-1])))[4:]
loglow1_2 = (np.log(np.array(low[1:]))-np.log(np.array(close[:-1])))[4:]
loglow1_3 = (np.log(np.array(low[1:]))-np.log(np.array(high[:-1])))[4:]
loglow1_4 = (np.log(np.array(low[1:]))-np.log(np.array(low[:-1])))[4:]

loglow2_1 = np.log(np.array(low[2:]))-np.log(np.array(open1[:-2]))
loglow2_2 = np.log(np.array(low[2:]))-np.log(np.array(close[:-2]))
loglow2_3 = np.log(np.array(low[2:]))-np.log(np.array(high[:-2]))
loglow2_4 = np.log(np.array(low[2:]))-np.log(np.array(low[:-2]))

loglow3_1 = np.log(np.array(low[3:]))-np.log(np.array(open1[:-3]))
loglow3_2 = np.log(np.array(low[3:]))-np.log(np.array(close[:-3]))
loglow3_3 = np.log(np.array(low[3:]))-np.log(np.array(high[:-3]))
loglow3_4 = np.log(np.array(low[3:]))-np.log(np.array(low[:-3]))

loglow4_1 = np.log(np.array(low[4:]))-np.log(np.array(open1[:-4]))
loglow4_2 = np.log(np.array(low[4:]))-np.log(np.array(close[:-4]))
loglow4_3 = np.log(np.array(low[4:]))-np.log(np.array(high[:-4]))
loglow4_4 = np.log(np.array(low[4:]))-np.log(np.array(low[:-4]))

loglow5_1 = np.log(np.array(low[5:]))-np.log(np.array(open1[:-5]))
loglow5_2 = np.log(np.array(low[5:]))-np.log(np.array(close[:-5]))
loglow5_3 = np.log(np.array(low[5:]))-np.log(np.array(high[:-5]))
loglow5_4 = np.log(np.array(low[5:]))-np.log(np.array(low[:-5]))



##################################################################################


close2 = df['close'][5:]
data = pd.DataFrame(columns=['close',
                             'logopen1_1','logopen1_2','logopen1_3','logopen1_4',
                             'logopen2_1','logopen2_2','logopen2_3','logopen2_4',
                             'logopen3_1','logopen3_2','logopen3_3','logopen3_4',
                             'logopen4_1','logopen4_2','logopen4_3','logopen4_4',
                             'logopen5_1','logopen5_2','logopen5_3','logopen5_4',
                             'logclose1_1','logclose1_2','logclose1_3','logclose1_4',
                             'logclose2_1','logclose2_2','logclose2_3','logclose2_4',
                             'logclose3_1','logclose3_2','logclose3_3','logclose3_4',
                             'logclose4_1','logclose4_2','logclose4_3','logclose4_4',
                             'logclose5_1','logclose5_2','logclose5_3','logclose5_4',
                             'loghigh1_1','loghigh1_2','loghigh1_3','loghigh1_4',
                             'loghigh2_1','loghigh2_2','loghigh2_3','loghigh2_4',
                             'loghigh3_1','loghigh3_2','loghigh3_3','loghigh3_4',
                             'loghigh4_1','loghigh4_2','loghigh4_3','loghigh4_4',
                             'loghigh5_1','loghigh5_2','loghigh5_3','loghigh5_4',
                             'loglow1_1','loglow1_2','loglow1_3','loglow1_4',
                             'loglow2_1','loglow2_2','loglow2_3','loglow2_4',
                             'loglow3_1','loglow3_2','loglow3_3','loglow3_4',
                             'loglow4_1','loglow4_2','loglow4_3','loglow4_4',
                             'loglow5_1','loglow5_2','loglow5_3','loglow5_4'])
                             
    
data['close'] = close2
data['logopen1_1'] = logopen1_1
data['logopen1_2'] = logopen1_2
data['logopen1_3'] = logopen1_3
data['logopen1_4'] = logopen1_4

data['logopen2_1'] = logopen2_1[3:]
data['logopen2_2'] = logopen2_2[3:]
data['logopen2_3'] = logopen2_3[3:]
data['logopen2_4'] = logopen2_4[3:]

data['logopen3_1'] = logopen3_1[2:]
data['logopen3_2'] = logopen3_2[2:]
data['logopen3_3'] = logopen3_3[2:]
data['logopen3_4'] = logopen3_4[2:]

data['logopen4_1'] = logopen4_1[1:]
data['logopen4_2'] = logopen4_2[1:]
data['logopen4_3'] = logopen4_3[1:]
data['logopen4_4'] = logopen4_4[1:]

data['logopen5_1'] = logopen5_1
data['logopen5_2'] = logopen5_2
data['logopen5_3'] = logopen5_3
data['logopen5_4'] = logopen5_4


data['logclose1_1'] = logclose1_1
data['logclose1_2'] = logclose1_2
data['logclose1_3'] = logclose1_3
data['logclose1_4'] = logclose1_4

data['logclose2_1'] = logclose2_1[3:]
data['logclose2_2'] = logclose2_2[3:]
data['logclose2_3'] = logclose2_3[3:]
data['logclose2_4'] = logclose2_4[3:]

data['logclose3_1'] = logclose3_1[2:]
data['logclose3_2'] = logclose3_2[2:]
data['logclose3_3'] = logclose3_3[2:]
data['logclose3_4'] = logclose3_4[2:]

data['logclose4_1'] = logclose4_1[1:]
data['logclose4_2'] = logclose4_2[1:]
data['logclose4_3'] = logclose4_3[1:]
data['logclose4_4'] = logclose4_4[1:]

data['logclose5_1'] = logclose5_1
data['logclose5_2'] = logclose5_2
data['logclose5_3'] = logclose5_3
data['logclose5_4'] = logclose5_4

data['loghigh1_1'] = loghigh1_1
data['loghigh1_2'] = loghigh1_2
data['loghigh1_3'] = loghigh1_3
data['loghigh1_4'] = loghigh1_4

data['loghigh2_1'] = loghigh2_1[3:]
data['loghigh2_2'] = loghigh2_2[3:]
data['loghigh2_3'] = loghigh2_3[3:]
data['loghigh2_4'] = loghigh2_4[3:]

data['loghigh3_1'] = loghigh3_1[2:]
data['loghigh3_2'] = loghigh3_2[2:]
data['loghigh3_3'] = loghigh3_3[2:]
data['loghigh3_4'] = loghigh3_4[2:]

data['loghigh4_1'] = loghigh4_1[1:]
data['loghigh4_2'] = loghigh4_2[1:]
data['loghigh4_3'] = loghigh4_3[1:]
data['loghigh4_4'] = loghigh4_4[1:]

data['loghigh5_1'] = loghigh5_1
data['loghigh5_2'] = loghigh5_2
data['loghigh5_3'] = loghigh5_3
data['loghigh5_4'] = loghigh5_4

data['loglow1_1'] = loglow1_1
data['loglow1_2'] = loglow1_2
data['loglow1_3'] = loglow1_3
data['loglow1_4'] = loglow1_4

data['loglow2_1'] = loglow2_1[3:]
data['loglow2_2'] = loglow2_2[3:]
data['loglow2_3'] = loglow2_3[3:]
data['loglow2_4'] = loglow2_4[3:]

data['loglow3_1'] = loglow3_1[2:]
data['loglow3_2'] = loglow3_2[2:]
data['loglow3_3'] = loglow3_3[2:]
data['loglow3_4'] = loglow3_4[2:]

data['loglow4_1'] = loglow4_1[1:]
data['loglow4_2'] = loglow4_2[1:]
data['loglow4_3'] = loglow4_3[1:]
data['loglow4_4'] = loglow4_4[1:]

data['loglow5_1'] = loglow5_1
data['loglow5_2'] = loglow5_2
data['loglow5_3'] = loglow5_3
data['loglow5_4'] = loglow5_4
    
    

data.to_csv(file_80,index = False)

#################################################################################

file_name = file_80
df1 = pd.read_csv(file_name)


    
a1 = df1['logopen1_1']
a2 = df1['logopen1_2']
a3 = df1['logopen1_3']
a4 = df1['logopen1_4']

a5 = df1['logopen2_1']
a6 = df1['logopen2_2']
a7 = df1['logopen2_3']
a8 = df1['logopen2_4']

a9 = df1['logopen3_1']
a10 = df1['logopen3_2']
a11 = df1['logopen3_3']
a12 = df1['logopen3_4'] 

a13 = df1['logopen4_1']
a14 = df1['logopen4_2']
a15 = df1['logopen4_3']
a16 = df1['logopen4_4']

a17 = df1['logopen5_1'] 
a18 = df1['logopen5_2'] 
a19 = df1['logopen5_3'] 
a20 = df1['logopen5_4'] 

a21 = df1['logclose1_1'] 
a22 = df1['logclose1_2'] 
a23 = df1['logclose1_3'] 
a24 = df1['logclose1_4'] 

a25 = df1['logclose2_1'] 
a26 = df1['logclose2_2'] 
a27 = df1['logclose2_3'] 
a28 = df1['logclose2_4'] 

a29 = df1['logclose3_1'] 
a30 = df1['logclose3_2'] 
a31 = df1['logclose3_3'] 
a32 = df1['logclose3_4'] 

a33 = df1['logclose4_1'] 
a34 = df1['logclose4_2'] 
a35 = df1['logclose4_3'] 
a36 = df1['logclose4_4'] 

a37 = df1['logclose5_1'] 
a38 = df1['logclose5_2'] 
a39 = df1['logclose5_3'] 
a40 = df1['logclose5_4'] 

a41 = df1['loghigh1_1'] 
a42 = df1['loghigh1_2'] 
a43 = df1['loghigh1_3'] 
a44 = df1['loghigh1_4'] 

a45 = df1['loghigh2_1'] 
a46 = df1['loghigh2_2'] 
a47 = df1['loghigh2_3'] 
a48 = df1['loghigh2_4'] 

a49 = df1['loghigh3_1'] 
a50 = df1['loghigh3_2'] 
a51 = df1['loghigh3_3']
a52 = df1['loghigh3_4'] 

a53 = df1['loghigh4_1'] 
a54 = df1['loghigh4_2'] 
a55 = df1['loghigh4_3'] 
a56 = df1['loghigh4_4'] 

a57 = df1['loghigh5_1'] 
a58 = df1['loghigh5_2'] 
a59 = df1['loghigh5_3'] 
a60 = df1['loghigh5_4'] 

a61 = df1['loglow1_1'] 
a62 = df1['loglow1_2'] 
a63 = df1['loglow1_3'] 
a64 = df1['loglow1_4'] 

a65 = df1['loglow2_1'] 
a66 = df1['loglow2_2'] 
a67 = df1['loglow2_3'] 
a68 = df1['loglow2_4'] 

a69 = df1['loglow3_1']
a70 = df1['loglow3_2'] 
a71 = df1['loglow3_3'] 
a72 = df1['loglow3_4'] 

a73 = df1['loglow4_1'] 
a74 = df1['loglow4_2'] 
a75 = df1['loglow4_3'] 
a76 = df1['loglow4_4'] 

a77 = df1['loglow5_1'] 
a78 = df1['loglow5_2']
a79 = df1['loglow5_3'] 
a80 = df1['loglow5_4'] 

    
file_name = file_Train
df2 = pd.read_csv(file_name)


# 5个80个特征，共400列 

long = Train + 30


a = 0
b = 1
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a1[j-i]
        b = b+80
    a = a+1
    b = 1

    
a = 0
b = 2
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a2[j-i]
        b = b+80
    a = a+1
    b = 2


a = 0
b = 3
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a3[j-i]
        b = b+80
    a = a+1
    b = 3
    
    
a = 0
b = 4
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a4[j-i]
        b = b+80
    a = a+1
    b = 4

a = 0
b = 5
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a5[j-i]
        b = b+80
    a = a+1
    b = 5
    
a = 0
b = 6
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a6[j-i]
        b = b+80
    a = a+1
    b = 6

a = 0
b = 7
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a7[j-i]
        b = b+80
    a = a+1
    b = 7
    
a = 0
b = 8
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a8[j-i]
        b = b+80
    a = a+1
    b = 8
    

a = 0
b = 9
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a9[j-i]
        b = b+80
    a = a+1
    b = 9
    
a = 0
b = 10
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a10[j-i]
        b = b+80
    a = a+1
    b = 10

a = 0
b = 11
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a11[j-i]
        b = b+80
    a = a+1
    b = 11
    
    
a = 0
b = 12
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a12[j-i]
        b = b+80
    a = a+1
    b = 12
    
a = 0
b = 13
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a13[j-i]
        b = b+80
    a = a+1
    b = 13
    
    
a = 0
b = 14
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a14[j-i]
        b = b+80
    a = a+1
    b = 14
    
    
    
a = 0
b = 15
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a15[j-i]
        b = b+80
    a = a+1
    b = 15
    

a = 0
b = 16
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a16[j-i]
        b = b+80
    a = a+1
    b = 16

    
a = 0
b = 17
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a17[j-i]
        b = b+80
    a = a+1
    b = 17


a = 0
b = 18
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a18[j-i]
        b = b+80
    a = a+1
    b = 18
    
    
a = 0
b = 19
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a19[j-i]
        b = b+80
    a = a+1
    b = 19

a = 0
b = 20
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a20[j-i]
        b = b+80
    a = a+1
    b = 20
    
a = 0
b = 21
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a21[j-i]
        b = b+80
    a = a+1
    b = 21

a = 0
b = 22
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a22[j-i]
        b = b+80
    a = a+1
    b = 22
    
a = 0
b = 23
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a23[j-i]
        b = b+80
    a = a+1
    b = 23
    

a = 0
b = 24
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a24[j-i]
        b = b+80
    a = a+1
    b = 24
    
a = 0
b = 25
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a25[j-i]
        b = b+80
    a = a+1
    b = 25

a = 0
b = 26
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a26[j-i]
        b = b+80
    a = a+1
    b = 26
    
    
a = 0
b = 27
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a27[j-i]
        b = b+80
    a = a+1
    b = 27
    
a = 0
b = 28
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a28[j-i]
        b = b+80
    a = a+1
    b = 28
    
    
a = 0
b = 29
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a29[j-i]
        b = b+80
    a = a+1
    b = 29
    
    
    
a = 0
b = 30
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a30[j-i]
        b = b+80
    a = a+1
    b = 30
    
    

a = 0
b = 31
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a31[j-i]
        b = b+80
    a = a+1
    b = 31

    
a = 0
b = 32
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a32[j-i]
        b = b+80
    a = a+1
    b = 32


a = 0
b = 33
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a33[j-i]
        b = b+80
    a = a+1
    b = 33
    
    
a = 0
b = 34
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a34[j-i]
        b = b+80
    a = a+1
    b = 34

a = 0
b = 35
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a35[j-i]
        b = b+80
    a = a+1
    b = 35
    
a = 0
b = 36
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a36[j-i]
        b = b+80
    a = a+1
    b = 36

a = 0
b = 37
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a37[j-i]
        b = b+80
    a = a+1
    b = 37
    
a = 0
b = 38
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a38[j-i]
        b = b+80
    a = a+1
    b = 38
    

a = 0
b = 39
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a39[j-i]
        b = b+80
    a = a+1
    b = 39
    
a = 0
b = 40
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a40[j-i]
        b = b+80
    a = a+1
    b = 40

a = 0
b = 41
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a41[j-i]
        b = b+80
    a = a+1
    b = 41
    
    
a = 0
b = 42
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a42[j-i]
        b = b+80
    a = a+1
    b = 42
    
a = 0
b = 43
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a43[j-i]
        b = b+80
    a = a+1
    b = 43
    
    
a = 0
b = 44
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a44[j-i]
        b = b+80
    a = a+1
    b = 44
    
    
    
a = 0
b = 45
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a45[j-i]
        b = b+80
    a = a+1
    b = 45
    

a = 0
b = 46
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a46[j-i]
        b = b+80
    a = a+1
    b = 46

    
a = 0
b = 47
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a47[j-i]
        b = b+80
    a = a+1
    b = 47


a = 0
b = 48
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a48[j-i]
        b = b+80
    a = a+1
    b = 48
    
    
a = 0
b = 49
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a49[j-i]
        b = b+80
    a = a+1
    b = 49

a = 0
b = 50
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a50[j-i]
        b = b+80
    a = a+1
    b = 50
    
a = 0
b = 51
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a51[j-i]
        b = b+80
    a = a+1
    b = 51

a = 0
b = 52
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a52[j-i]
        b = b+80
    a = a+1
    b = 52
    
a = 0
b = 53
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a53[j-i]
        b = b+80
    a = a+1
    b = 53
    

a = 0
b = 54
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a54[j-i]
        b = b+80
    a = a+1
    b = 54
    
a = 0
b = 55
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a55[j-i]
        b = b+80
    a = a+1
    b = 55

a = 0
b = 56
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a56[j-i]
        b = b+80
    a = a+1
    b = 56
    
    
a = 0
b = 57
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a57[j-i]
        b = b+80
    a = a+1
    b = 57
    
a = 0
b = 58
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a58[j-i]
        b = b+80
    a = a+1
    b = 58
    
    
a = 0
b = 59
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a59[j-i]
        b = b+80
    a = a+1
    b = 59
    
    
    
a = 0
b = 60
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a60[j-i]
        b = b+80
    a = a+1
    b = 60
    

a = 0
b = 61
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a61[j-i]
        b = b+80
    a = a+1
    b = 61

    
a = 0
b = 62
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a62[j-i]
        b = b+80
    a = a+1
    b = 62


a = 0
b = 63
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a63[j-i]
        b = b+80
    a = a+1
    b = 63
    
    
a = 0
b = 64
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a64[j-i]
        b = b+80
    a = a+1
    b = 64

a = 0
b = 65
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a65[j-i]
        b = b+80
    a = a+1
    b = 65

    
a = 0
b = 66
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a66[j-i]
        b = b+80
    a = a+1
    b = 66

a = 0
b = 67
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a67[j-i]
        b = b+80
    a = a+1
    b = 67
    
a = 0
b = 68
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a68[j-i]
        b = b+80
    a = a+1
    b = 68
    

a = 0
b = 69
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a69[j-i]
        b = b+80
    a = a+1
    b = 69
    
a = 0
b = 70
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a70[j-i]
        b = b+80
    a = a+1
    b = 70

a = 0
b = 71
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a71[j-i]
        b = b+80
    a = a+1
    b = 71
    
    
a = 0
b = 72
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a72[j-i]
        b = b+80
    a = a+1
    b = 72
    
a = 0
b = 73
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a73[j-i]
        b = b+80
    a = a+1
    b = 73
    
    
a = 0
b = 74
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a74[j-i]
        b = b+80
    a = a+1
    b = 74
    
    
    
a = 0
b = 75
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a75[j-i]
        b = b+80
    a = a+1
    b = 75
    

a = 0
b = 76
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a76[j-i]
        b = b+80
    a = a+1
    b = 76

    
a = 0
b = 77
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a77[j-i]
        b = b+80
    a = a+1
    b = 77


a = 0
b = 78
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a78[j-i]
        b = b+80
    a = a+1
    b = 78
    
    
a = 0
b = 79
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a79[j-i]
        b = b+80
    a = a+1
    b = 79

a = 0
b = 80
for j in range(30,long):   
    for i in range(5):
        df2.iloc[a,b]=a80[j-i]
        b = b+80
    a = a+1
    b = 80
    
    
    
df2.to_csv(file_400_Train,index = False)
print('Train file OK')
##############################################################################



file_name = file_80
df1 = pd.read_csv(file_name)


    
a1 = df1['logopen1_1']
a2 = df1['logopen1_2']
a3 = df1['logopen1_3']
a4 = df1['logopen1_4']

a5 = df1['logopen2_1']
a6 = df1['logopen2_2']
a7 = df1['logopen2_3']
a8 = df1['logopen2_4']

a9 = df1['logopen3_1']
a10 = df1['logopen3_2']
a11 = df1['logopen3_3']
a12 = df1['logopen3_4'] 

a13 = df1['logopen4_1']
a14 = df1['logopen4_2']
a15 = df1['logopen4_3']
a16 = df1['logopen4_4']

a17 = df1['logopen5_1'] 
a18 = df1['logopen5_2'] 
a19 = df1['logopen5_3'] 
a20 = df1['logopen5_4'] 

a21 = df1['logclose1_1'] 
a22 = df1['logclose1_2'] 
a23 = df1['logclose1_3'] 
a24 = df1['logclose1_4'] 

a25 = df1['logclose2_1'] 
a26 = df1['logclose2_2'] 
a27 = df1['logclose2_3'] 
a28 = df1['logclose2_4'] 

a29 = df1['logclose3_1'] 
a30 = df1['logclose3_2'] 
a31 = df1['logclose3_3'] 
a32 = df1['logclose3_4'] 

a33 = df1['logclose4_1'] 
a34 = df1['logclose4_2'] 
a35 = df1['logclose4_3'] 
a36 = df1['logclose4_4'] 

a37 = df1['logclose5_1'] 
a38 = df1['logclose5_2'] 
a39 = df1['logclose5_3'] 
a40 = df1['logclose5_4'] 

a41 = df1['loghigh1_1'] 
a42 = df1['loghigh1_2'] 
a43 = df1['loghigh1_3'] 
a44 = df1['loghigh1_4'] 

a45 = df1['loghigh2_1'] 
a46 = df1['loghigh2_2'] 
a47 = df1['loghigh2_3'] 
a48 = df1['loghigh2_4'] 

a49 = df1['loghigh3_1'] 
a50 = df1['loghigh3_2'] 
a51 = df1['loghigh3_3']
a52 = df1['loghigh3_4'] 

a53 = df1['loghigh4_1'] 
a54 = df1['loghigh4_2'] 
a55 = df1['loghigh4_3'] 
a56 = df1['loghigh4_4'] 

a57 = df1['loghigh5_1'] 
a58 = df1['loghigh5_2'] 
a59 = df1['loghigh5_3'] 
a60 = df1['loghigh5_4'] 

a61 = df1['loglow1_1'] 
a62 = df1['loglow1_2'] 
a63 = df1['loglow1_3'] 
a64 = df1['loglow1_4'] 

a65 = df1['loglow2_1'] 
a66 = df1['loglow2_2'] 
a67 = df1['loglow2_3'] 
a68 = df1['loglow2_4'] 

a69 = df1['loglow3_1']
a70 = df1['loglow3_2'] 
a71 = df1['loglow3_3'] 
a72 = df1['loglow3_4'] 

a73 = df1['loglow4_1'] 
a74 = df1['loglow4_2'] 
a75 = df1['loglow4_3'] 
a76 = df1['loglow4_4'] 

a77 = df1['loglow5_1'] 
a78 = df1['loglow5_2']
a79 = df1['loglow5_3'] 
a80 = df1['loglow5_4'] 

    
file_name = file_Test
df2 = pd.read_csv(file_name)




long = len(df1)-1
short = long - Test


a = 0
b = 1
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a1[j-i]
        b = b+80
    a = a+1
    b = 1

    
a = 0
b = 2
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a2[j-i]
        b = b+80
    a = a+1
    b = 2


a = 0
b = 3
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a3[j-i]
        b = b+80
    a = a+1
    b = 3
    
    
a = 0
b = 4
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a4[j-i]
        b = b+80
    a = a+1
    b = 4

a = 0
b = 5
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a5[j-i]
        b = b+80
    a = a+1
    b = 5
    
a = 0
b = 6
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a6[j-i]
        b = b+80
    a = a+1
    b = 6

a = 0
b = 7
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a7[j-i]
        b = b+80
    a = a+1
    b = 7
    
a = 0
b = 8
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a8[j-i]
        b = b+80
    a = a+1
    b = 8
    

a = 0
b = 9
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a9[j-i]
        b = b+80
    a = a+1
    b = 9
    
a = 0
b = 10
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a10[j-i]
        b = b+80
    a = a+1
    b = 10

a = 0
b = 11
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a11[j-i]
        b = b+80
    a = a+1
    b = 11
    
    
a = 0
b = 12
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a12[j-i]
        b = b+80
    a = a+1
    b = 12
    
a = 0
b = 13
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a13[j-i]
        b = b+80
    a = a+1
    b = 13
    
    
a = 0
b = 14
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a14[j-i]
        b = b+80
    a = a+1
    b = 14
    
    
    
a = 0
b = 15
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a15[j-i]
        b = b+80
    a = a+1
    b = 15
    

a = 0
b = 16
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a16[j-i]
        b = b+80
    a = a+1
    b = 16

    
a = 0
b = 17
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a17[j-i]
        b = b+80
    a = a+1
    b = 17


a = 0
b = 18
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a18[j-i]
        b = b+80
    a = a+1
    b = 18
    
    
a = 0
b = 19
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a19[j-i]
        b = b+80
    a = a+1
    b = 19

a = 0
b = 20
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a20[j-i]
        b = b+80
    a = a+1
    b = 20
    
a = 0
b = 21
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a21[j-i]
        b = b+80
    a = a+1
    b = 21

a = 0
b = 22
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a22[j-i]
        b = b+80
    a = a+1
    b = 22
    
a = 0
b = 23
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a23[j-i]
        b = b+80
    a = a+1
    b = 23
    

a = 0
b = 24
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a24[j-i]
        b = b+80
    a = a+1
    b = 24
    
a = 0
b = 25
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a25[j-i]
        b = b+80
    a = a+1
    b = 25

a = 0
b = 26
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a26[j-i]
        b = b+80
    a = a+1
    b = 26
    
    
a = 0
b = 27
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a27[j-i]
        b = b+80
    a = a+1
    b = 27
    
a = 0
b = 28
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a28[j-i]
        b = b+80
    a = a+1
    b = 28
    
    
a = 0
b = 29
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a29[j-i]
        b = b+80
    a = a+1
    b = 29
    
    
    
a = 0
b = 30
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a30[j-i]
        b = b+80
    a = a+1
    b = 30
    
    

a = 0
b = 31
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a31[j-i]
        b = b+80
    a = a+1
    b = 31

    
a = 0
b = 32
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a32[j-i]
        b = b+80
    a = a+1
    b = 32


a = 0
b = 33
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a33[j-i]
        b = b+80
    a = a+1
    b = 33
    
    
a = 0
b = 34
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a34[j-i]
        b = b+80
    a = a+1
    b = 34

a = 0
b = 35
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a35[j-i]
        b = b+80
    a = a+1
    b = 35
    
a = 0
b = 36
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a36[j-i]
        b = b+80
    a = a+1
    b = 36

a = 0
b = 37
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a37[j-i]
        b = b+80
    a = a+1
    b = 37
    
a = 0
b = 38
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a38[j-i]
        b = b+80
    a = a+1
    b = 38
    

a = 0
b = 39
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a39[j-i]
        b = b+80
    a = a+1
    b = 39
    
a = 0
b = 40
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a40[j-i]
        b = b+80
    a = a+1
    b = 40

a = 0
b = 41
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a41[j-i]
        b = b+80
    a = a+1
    b = 41
    
    
a = 0
b = 42
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a42[j-i]
        b = b+80
    a = a+1
    b = 42
    
a = 0
b = 43
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a43[j-i]
        b = b+80
    a = a+1
    b = 43
    
    
a = 0
b = 44
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a44[j-i]
        b = b+80
    a = a+1
    b = 44
    
    
    
a = 0
b = 45
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a45[j-i]
        b = b+80
    a = a+1
    b = 45
    

a = 0
b = 46
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a46[j-i]
        b = b+80
    a = a+1
    b = 46

    
a = 0
b = 47
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a47[j-i]
        b = b+80
    a = a+1
    b = 47


a = 0
b = 48
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a48[j-i]
        b = b+80
    a = a+1
    b = 48
    
    
a = 0
b = 49
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a49[j-i]
        b = b+80
    a = a+1
    b = 49

a = 0
b = 50
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a50[j-i]
        b = b+80
    a = a+1
    b = 50
    
a = 0
b = 51
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a51[j-i]
        b = b+80
    a = a+1
    b = 51

a = 0
b = 52
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a52[j-i]
        b = b+80
    a = a+1
    b = 52
    
a = 0
b = 53
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a53[j-i]
        b = b+80
    a = a+1
    b = 53
    

a = 0
b = 54
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a54[j-i]
        b = b+80
    a = a+1
    b = 54
    
a = 0
b = 55
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a55[j-i]
        b = b+80
    a = a+1
    b = 55

a = 0
b = 56
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a56[j-i]
        b = b+80
    a = a+1
    b = 56
    
    
a = 0
b = 57
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a57[j-i]
        b = b+80
    a = a+1
    b = 57
    
a = 0
b = 58
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a58[j-i]
        b = b+80
    a = a+1
    b = 58
    
    
a = 0
b = 59
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a59[j-i]
        b = b+80
    a = a+1
    b = 59
    
    
    
a = 0
b = 60
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a60[j-i]
        b = b+80
    a = a+1
    b = 60
    

a = 0
b = 61
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a61[j-i]
        b = b+80
    a = a+1
    b = 61

    
a = 0
b = 62
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a62[j-i]
        b = b+80
    a = a+1
    b = 62


a = 0
b = 63
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a63[j-i]
        b = b+80
    a = a+1
    b = 63
    
    
a = 0
b = 64
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a64[j-i]
        b = b+80
    a = a+1
    b = 64

a = 0
b = 65
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a65[j-i]
        b = b+80
    a = a+1
    b = 65

    
a = 0
b = 66
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a66[j-i]
        b = b+80
    a = a+1
    b = 66

a = 0
b = 67
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a67[j-i]
        b = b+80
    a = a+1
    b = 67
    
a = 0
b = 68
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a68[j-i]
        b = b+80
    a = a+1
    b = 68
    

a = 0
b = 69
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a69[j-i]
        b = b+80
    a = a+1
    b = 69
    
a = 0
b = 70
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a70[j-i]
        b = b+80
    a = a+1
    b = 70

a = 0
b = 71
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a71[j-i]
        b = b+80
    a = a+1
    b = 71
    
    
a = 0
b = 72
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a72[j-i]
        b = b+80
    a = a+1
    b = 72
    
a = 0
b = 73
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a73[j-i]
        b = b+80
    a = a+1
    b = 73
    
    
a = 0
b = 74
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a74[j-i]
        b = b+80
    a = a+1
    b = 74
    
    
    
a = 0
b = 75
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a75[j-i]
        b = b+80
    a = a+1
    b = 75
    

a = 0
b = 76
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a76[j-i]
        b = b+80
    a = a+1
    b = 76

    
a = 0
b = 77
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a77[j-i]
        b = b+80
    a = a+1
    b = 77


a = 0
b = 78
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a78[j-i]
        b = b+80
    a = a+1
    b = 78
    
    
a = 0
b = 79
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a79[j-i]
        b = b+80
    a = a+1
    b = 79

a = 0
b = 80
for j in range(short,long):   
    for i in range(5):
        df2.iloc[a,b]=a80[j-i]
        b = b+80
    a = a+1
    b = 80

    
df2 = df2.drop(columns='A0')  # 删除A0这一列 
df2.to_csv(file_400_Test,index = False)
print('Test file OK')


###########################################################################################
#复制训练集，加到测试的前面和后面，为了应对PCA降维

data_Train = pd.read_csv(file_400_Train)
data_Train = data_Train.drop('A0', axis=1)   ### 删除A0这列

data = pd.concat([data_Train, df2], ignore_index=True)
data = pd.concat([data, data_Train], ignore_index=True)

data.to_csv(file_400_Train_3,index = False)
print('Train + Test PCA file OK')







###########################################################################################

file_name = file_400_Train
df_400 = pd.read_csv(file_name)

if df_400.iloc[0,22] == df_label.iloc[30,1]:   ### 如何标签9是26  8是27，7是28，6是29，5是30,15是20
    for i in range(31,Train+31):
        df_400.iloc[i-31,0] = df_label.iloc[i,2]
    df_400.to_csv(file_400_Train,index = False)
    print('Label OK')
    
else:
    print('Label NG !!!')
    


