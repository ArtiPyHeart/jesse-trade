import rqdatac
from rqdatac import LiveMarketDataClient
rqdatac.init()
from datetime import datetime, date
import numpy as np
import pandas as pd
from pycaret.classification import *
import datetime
import winsound
from matplotlib import pyplot as plt


#################################################################################################################
# 下载1分钟数据
#################################################################################################################

def predict():
    
    global close_1m


    if auto_trading == 0:

        y_mode = int(datetime.datetime.now().strftime('%Y'))
        m_mode = int(datetime.datetime.now().strftime('%m'))
        day_mode = int(datetime.datetime.now().strftime('%d'))


        print("mode=0，白天   mode=1，平日晚上   mode=3，周五晚上   mode=4，自定义")
        mode = int(input("mode = "))

        if mode == 0:
            print('*** 白天 OK ***')

        if mode == 1:
            day_mode = day_mode + 1
            print('*** 平日晚上 OK ***')

        if mode == 3:
            day_mode = day_mode + 3
            print('*** 周五晚上 OK ***')

        if mode == 4:
            y_in = int(input("年 = "))
            m_in = int(input("月 = "))
            d_in = int(input("日 = "))
            y_mode = y_in
            m_mode = m_in
            day_mode = d_in
            print('*** 自定义 OK ***')


    if auto_trading == 1:

        date_f = pd.read_csv('date.csv')
        now_time = datetime.datetime.now()
        hour_time = now_time.hour

        if hour_time < 20:
            y_mode = int(datetime.datetime.now().strftime('%Y'))
            m_mode = int(datetime.datetime.now().strftime('%m'))
            day_mode = int(datetime.datetime.now().strftime('%d'))


        if hour_time >= 20:

            y_mode = int(datetime.datetime.now().strftime('%Y'))
            m_mode = int(datetime.datetime.now().strftime('%m'))
            day_mode = int(datetime.datetime.now().strftime('%d'))

            filtered_rows = date_f[(date_f['A0'] == int(str(y_mode)+str(m_mode)+str(day_mode)))]

            
            y_mode = filtered_rows['A1'].iloc[0]
            m_mode = filtered_rows['A2'].iloc[0]
            day_mode = filtered_rows['A3'].iloc[0]



    data_1m = rqdatac.get_price('RB99', start_date='2024-10-08', end_date=str(y_mode)+'-'+str(m_mode)+'-'+str(day_mode), frequency='1m')
    data_1m.to_csv('data_1m.csv')
    
    print('data_1m_____OK')
    
    close_1m = data_1m['close'].iloc[-1]


    

    #################################################################################################################
    # 生成成交额的Bar
    #################################################################################################################

    file_name = 'data_1m.csv'
    df = pd.read_csv(file_name)

    #### 时间轴对齐，与2010年开始是一致的，减去多余的数据

    for i in range(1):  # 减少多少行
        df = df.drop([i])
    df = df.reset_index(drop=True)



    size = 31000

    buf_o = []
    buf_h = []
    buf_l = []
    buf_c = []
    buf_v = []
    res = []
    pos_sum = 0


    for i in range(0,len(df)):
        
        p_t = df['datetime'].iloc[i]

        p_o = df['open'].iloc[i]
        p_h = df['high'].iloc[i]
        p_l = df['low'].iloc[i]
        p_c = df['close'].iloc[i]
        p_v = df['volume'].iloc[i]
        p_tt = df['total_turnover'].iloc[i]



        di = df.index.values[i]


        buf_o.append(p_o)
        buf_h.append(p_h)
        buf_l.append(p_l)
        buf_c.append(p_c)
        buf_v.append(p_v)


        pos_sum = pos_sum + (p_tt / 1000000)

        if pos_sum >= size:
            o = buf_o[0]
            h = max(buf_h)
            l = min(buf_l)
            c = buf_c[-1]
            v = sum(buf_v)
            p = pos_sum

            

            res.append({
                'eob': p_t,
                'open': o,
                'high': h,
                'low': l,
                'close': c,  
                'volume': v,
                'pos': p,
                'hang':di
            })

            buf_o = []
            buf_h = []
            buf_l = []
            buf_c = []
            buf_v = []
            pos_sum = 0
            
            
            
            


    data_bar = pd.DataFrame(res).set_index('eob')
    data_bar.to_csv("data_bar.csv")
    
    print('data_bar____OK')



    ### 将剩余的pos_sum记录下来
    
    percent = pos_sum / 31000 * 100

    #################################################################################################################
    # 追加最新的成交额加入到文件，等到满足条件了，就开始运行模型
    #################################################################################################################

    new_price = [{'price': pos_sum}]  # 做一个新表，把上面满足条件之后剩余的成交额累加，存入新表
    new_volume_file = pd.DataFrame(new_price)
    new_volume_file.to_csv("new_volume_file.csv", index=False)

    #################################################################################################################
    # 特征处理79，运行模型，查看上一个190的方向与概率
    #################################################################################################################

    file_name = 'data_bar.csv'
    df = pd.read_csv(file_name)

   
    data_error = 0   
    
    #################################################################################################################
    

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

    # 重置data的索引值，从0开始，同时删除原索引值
    data = data.reset_index(drop=True)

    a1 = data['logopen1_1']
    a2 = data['logopen1_2']
    a3 = data['logopen1_3']
    a4 = data['logopen1_4']

    a5 = data['logopen2_1']
    a6 = data['logopen2_2']
    a7 = data['logopen2_3']
    a8 = data['logopen2_4']

    a9 = data['logopen3_1']
    a10 = data['logopen3_2']
    a11 = data['logopen3_3']
    a12 = data['logopen3_4'] 

    a13 = data['logopen4_1']
    a14 = data['logopen4_2']
    a15 = data['logopen4_3']
    a16 = data['logopen4_4']

    a17 = data['logopen5_1'] 
    a18 = data['logopen5_2'] 
    a19 = data['logopen5_3'] 
    a20 = data['logopen5_4'] 

    a21 = data['logclose1_1'] 
    a22 = data['logclose1_2'] 
    a23 = data['logclose1_3'] 
    a24 = data['logclose1_4'] 

    a25 = data['logclose2_1'] 
    a26 = data['logclose2_2'] 
    a27 = data['logclose2_3'] 
    a28 = data['logclose2_4'] 

    a29 = data['logclose3_1'] 
    a30 = data['logclose3_2'] 
    a31 = data['logclose3_3'] 
    a32 = data['logclose3_4'] 

    a33 = data['logclose4_1'] 
    a34 = data['logclose4_2'] 
    a35 = data['logclose4_3'] 
    a36 = data['logclose4_4'] 

    a37 = data['logclose5_1'] 
    a38 = data['logclose5_2'] 
    a39 = data['logclose5_3'] 
    a40 = data['logclose5_4'] 

    a41 = data['loghigh1_1'] 
    a42 = data['loghigh1_2'] 
    a43 = data['loghigh1_3'] 
    a44 = data['loghigh1_4'] 

    a45 = data['loghigh2_1'] 
    a46 = data['loghigh2_2'] 
    a47 = data['loghigh2_3'] 
    a48 = data['loghigh2_4'] 

    a49 = data['loghigh3_1'] 
    a50 = data['loghigh3_2'] 
    a51 = data['loghigh3_3']
    a52 = data['loghigh3_4'] 

    a53 = data['loghigh4_1'] 
    a54 = data['loghigh4_2'] 
    a55 = data['loghigh4_3'] 
    a56 = data['loghigh4_4'] 

    a57 = data['loghigh5_1'] 
    a58 = data['loghigh5_2'] 
    a59 = data['loghigh5_3'] 
    a60 = data['loghigh5_4'] 

    a61 = data['loglow1_1'] 
    a62 = data['loglow1_2'] 
    a63 = data['loglow1_3'] 
    a64 = data['loglow1_4'] 

    a65 = data['loglow2_1'] 
    a66 = data['loglow2_2'] 
    a67 = data['loglow2_3'] 
    a68 = data['loglow2_4'] 

    a69 = data['loglow3_1']
    a70 = data['loglow3_2'] 
    a71 = data['loglow3_3'] 
    a72 = data['loglow3_4'] 

    a73 = data['loglow4_1'] 
    a74 = data['loglow4_2'] 
    a75 = data['loglow4_3'] 
    a76 = data['loglow4_4'] 

    a77 = data['loglow5_1'] 
    a78 = data['loglow5_2']
    a79 = data['loglow5_3'] 
    a80 = data['loglow5_4'] 
 
    df = pd.read_csv('data_bar.csv')

    line = df[df['hang'] == 26711]
        # 计算到底部的行数
    line_num = len(df) - line.index[-1] - 1



    file_name ='1_400.csv'   
    file_temp = pd.read_csv(file_name)
    ###### 复制第0行N遍 ######
    temp_df = file_temp.iloc[0]
    for t in range(line_num-1): 
            file_temp.loc[file_temp.shape[0]] = temp_df
    # file_temp.to_csv(str(line_num) + ".csv",index = False)

    df2 = file_temp




    # 5个80个特征

    ddd = len(data)

    a = 0
    b = 1
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a1[j - i]
            b = b + 80
        a = a + 1
        b = 1

    a = 0
    b = 2
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a2[j - i]
            b = b + 80
        a = a + 1
        b = 2

    a = 0
    b = 3
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a3[j - i]
            b = b + 80
        a = a + 1
        b = 3

    a = 0
    b = 4
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a4[j - i]
            b = b + 80
        a = a + 1
        b = 4

    a = 0
    b = 5
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a5[j - i]
            b = b + 80
        a = a + 1
        b = 5

    a = 0
    b = 6
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a6[j - i]
            b = b + 80
        a = a + 1
        b = 6

    a = 0
    b = 7
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a7[j - i]
            b = b + 80
        a = a + 1
        b = 7

    a = 0
    b = 8
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a8[j - i]
            b = b + 80
        a = a + 1
        b = 8

    a = 0
    b = 9
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a9[j - i]
            b = b + 80
        a = a + 1
        b = 9

    a = 0
    b = 10
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a10[j - i]
            b = b + 80
        a = a + 1
        b = 10

    a = 0
    b = 11
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a11[j - i]
            b = b + 80
        a = a + 1
        b = 11

    a = 0
    b = 12
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a12[j - i]
            b = b + 80
        a = a + 1
        b = 12

    a = 0
    b = 13
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a13[j - i]
            b = b + 80
        a = a + 1
        b = 13

    a = 0
    b = 14
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a14[j - i]
            b = b + 80
        a = a + 1
        b = 14

    a = 0
    b = 15
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a15[j - i]
            b = b + 80
        a = a + 1
        b = 15

    a = 0
    b = 16
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a16[j - i]
            b = b + 80
        a = a + 1
        b = 16

    a = 0
    b = 17
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a17[j - i]
            b = b + 80
        a = a + 1
        b = 17

    a = 0
    b = 18
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a18[j - i]
            b = b + 80
        a = a + 1
        b = 18

    a = 0
    b = 19
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a19[j - i]
            b = b + 80
        a = a + 1
        b = 19

    a = 0
    b = 20
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a20[j - i]
            b = b + 80
        a = a + 1
        b = 20

    a = 0
    b = 21
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a21[j - i]
            b = b + 80
        a = a + 1
        b = 21

    a = 0
    b = 22
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a22[j - i]
            b = b + 80
        a = a + 1
        b = 22

    a = 0
    b = 23
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a23[j - i]
            b = b + 80
        a = a + 1
        b = 23

    a = 0
    b = 24
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a24[j - i]
            b = b + 80
        a = a + 1
        b = 24

    a = 0
    b = 25
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a25[j - i]
            b = b + 80
        a = a + 1
        b = 25

    a = 0
    b = 26
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a26[j - i]
            b = b + 80
        a = a + 1
        b = 26

    a = 0
    b = 27
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a27[j - i]
            b = b + 80
        a = a + 1
        b = 27

    a = 0
    b = 28
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a28[j - i]
            b = b + 80
        a = a + 1
        b = 28

    a = 0
    b = 29
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a29[j - i]
            b = b + 80
        a = a + 1
        b = 29

    a = 0
    b = 30
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a30[j - i]
            b = b + 80
        a = a + 1
        b = 30

    a = 0
    b = 31
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a31[j - i]
            b = b + 80
        a = a + 1
        b = 31

    a = 0
    b = 32
    for j in range(ddd-line_num,ddd):
        for i in range(5):
            df2.iloc[a, b] = a32[j - i]
            b = b + 80
        a = a + 1
        b = 32

    a = 0
    b = 33
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a33[j-i]
            b = b+80
        a = a+1
        b = 33
        
        
    a = 0
    b = 34
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a34[j-i]
            b = b+80
        a = a+1
        b = 34

    a = 0
    b = 35
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a35[j-i]
            b = b+80
        a = a+1
        b = 35
        
    a = 0
    b = 36
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a36[j-i]
            b = b+80
        a = a+1
        b = 36

    a = 0
    b = 37
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a37[j-i]
            b = b+80
        a = a+1
        b = 37
        
    a = 0
    b = 38
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a38[j-i]
            b = b+80
        a = a+1
        b = 38
        

    a = 0
    b = 39
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a39[j-i]
            b = b+80
        a = a+1
        b = 39
        
    a = 0
    b = 40
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a40[j-i]
            b = b+80
        a = a+1
        b = 40

    a = 0
    b = 41
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a41[j-i]
            b = b+80
        a = a+1
        b = 41
        
        
    a = 0
    b = 42
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a42[j-i]
            b = b+80
        a = a+1
        b = 42
        
    a = 0
    b = 43
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a43[j-i]
            b = b+80
        a = a+1
        b = 43
        
        
    a = 0
    b = 44
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a44[j-i]
            b = b+80
        a = a+1
        b = 44
        
        
        
    a = 0
    b = 45
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a45[j-i]
            b = b+80
        a = a+1
        b = 45
        

    a = 0
    b = 46
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a46[j-i]
            b = b+80
        a = a+1
        b = 46

        
    a = 0
    b = 47
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a47[j-i]
            b = b+80
        a = a+1
        b = 47


    a = 0
    b = 48
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a48[j-i]
            b = b+80
        a = a+1
        b = 48
        
        
    a = 0
    b = 49
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a49[j-i]
            b = b+80
        a = a+1
        b = 49

    a = 0
    b = 50
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a50[j-i]
            b = b+80
        a = a+1
        b = 50
        
    a = 0
    b = 51
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a51[j-i]
            b = b+80
        a = a+1
        b = 51

    a = 0
    b = 52
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a52[j-i]
            b = b+80
        a = a+1
        b = 52
        
    a = 0
    b = 53
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a53[j-i]
            b = b+80
        a = a+1
        b = 53
        

    a = 0
    b = 54
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a54[j-i]
            b = b+80
        a = a+1
        b = 54
        
    a = 0
    b = 55
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a55[j-i]
            b = b+80
        a = a+1
        b = 55

    a = 0
    b = 56
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a56[j-i]
            b = b+80
        a = a+1
        b = 56
        
        
    a = 0
    b = 57
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a57[j-i]
            b = b+80
        a = a+1
        b = 57
        
    a = 0
    b = 58
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a58[j-i]
            b = b+80
        a = a+1
        b = 58
        
        
    a = 0
    b = 59
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a59[j-i]
            b = b+80
        a = a+1
        b = 59
        
        
        
    a = 0
    b = 60
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a60[j-i]
            b = b+80
        a = a+1
        b = 60
        

    a = 0
    b = 61
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a61[j-i]
            b = b+80
        a = a+1
        b = 61

        
    a = 0
    b = 62
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a62[j-i]
            b = b+80
        a = a+1
        b = 62


    a = 0
    b = 63
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a63[j-i]
            b = b+80
        a = a+1
        b = 63
        
        
    a = 0
    b = 64
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a64[j-i]
            b = b+80
        a = a+1
        b = 64

    a = 0
    b = 65
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a65[j-i]
            b = b+80
        a = a+1
        b = 65

        
    a = 0
    b = 66
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a66[j-i]
            b = b+80
        a = a+1
        b = 66

    a = 0
    b = 67
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a67[j-i]
            b = b+80
        a = a+1
        b = 67
        
    a = 0
    b = 68
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a68[j-i]
            b = b+80
        a = a+1
        b = 68
        

    a = 0
    b = 69
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a69[j-i]
            b = b+80
        a = a+1
        b = 69
        
    a = 0
    b = 70
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a70[j-i]
            b = b+80
        a = a+1
        b = 70

    a = 0
    b = 71
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a71[j-i]
            b = b+80
        a = a+1
        b = 71
        
        
    a = 0
    b = 72
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a72[j-i]
            b = b+80
        a = a+1
        b = 72
        
    a = 0
    b = 73
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a73[j-i]
            b = b+80
        a = a+1
        b = 73
        
        
    a = 0
    b = 74
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a74[j-i]
            b = b+80
        a = a+1
        b = 74
        
        
        
    a = 0
    b = 75
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a75[j-i]
            b = b+80
        a = a+1
        b = 75
        

    a = 0
    b = 76
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a76[j-i]
            b = b+80
        a = a+1
        b = 76

        
    a = 0
    b = 77
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a77[j-i]
            b = b+80
        a = a+1
        b = 77


    a = 0
    b = 78
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a78[j-i]
            b = b+80
        a = a+1
        b = 78
        
        
    a = 0
    b = 79
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a79[j-i]
            b = b+80
        a = a+1
        b = 79

    a = 0
    b = 80
    for j in range(ddd-line_num,ddd):   
        for i in range(5):
            df2.iloc[a,b]=a80[j-i]
            b = b+80
        a = a+1
        b = 80


    df2 = df2.drop(columns='A0')  # 删除A0这一列
    df2.to_csv("data_400.csv", index=False)
    
    print('data_400____OK')



 




   ###########################################################################################################
    data_1x = pd.read_csv('Train_10877.csv')
    data_1 = pd.read_csv('Test_1213.csv')

    data_m1 = pd.read_csv('data_400.csv')

    data_11 = pd.concat([data_1x, data_1], ignore_index=True)
    data_11 = pd.concat([data_11, data_m1], ignore_index=True)

    data_1 = pd.concat([data_11, data_1x], ignore_index=True)

    ###########################################################################################################

    model_1 = load_model('./m/613_6x')
    model_2 = load_model('./m/913_228x')



    predictions_1 = predict_model(model_1, data=data_1)
    predictions_2 = predict_model(model_2, data=data_1)



    predictions_1.to_csv("m1.csv", index=False)
    predictions_2.to_csv("m2.csv", index=False)



    preds_1 = predictions_1['prediction_label'].iloc[-10878]
    score_1 = predictions_1['prediction_score'].iloc[-10878]

    preds_2 = predictions_2['prediction_label'].iloc[-10878]
    score_2 = predictions_2['prediction_score'].iloc[-10878]



    if preds_1 == preds_2:

        with open('RB_A.txt', 'w') as file:
            file.write(str(preds_1))

    else:

        with open('RB_A.txt', 'w') as file:
            file.write('2')




    if preds_1 == 1:
        preds_1 = 'Buy'
    else:
        preds_1 = 'Sell'

    if preds_2 == 1:
        preds_2 = 'Buy'
    else:
        preds_2 = 'Sell'



    print('++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++++++++++++++++')
    print('Time: ', df['eob'].values[-1])
    print('++++++++++++++++++++++++++++++++++++++++')
    print('Model_1: ' + preds_1 + ' ' + str(score_1))
    print('Model_2: ' + preds_2 + ' ' + str(score_2))
    print('++++++++++++++++++++++++++++++++++++++++')
    print('New Price Sum: ', str(round(pos_sum,2)))
    print('Percent: ', str(round(percent,2))+'%')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')




    return(data_error)


#################################################################################################################


def start_A():

    
    while True:        
        data_err = predict()
        if data_err == 0:
            break


#####################################################################################################################

def start_size():
    
    global close_1m

    client = LiveMarketDataClient()
    client.subscribe('bar_RB99_1m')
    

    for market in client.listen():
        
        time_str = str(market["datetime"])[0:4]+'-'+str(market["datetime"])[4:6]+'-'+str(market["datetime"])[6:8]+' '+str(market["datetime"])[8:10]+':'+str(market["datetime"])[10:12]
        print('++++++++++++++++++++++++++++++++++++++++')
        print("1m Turnover: ", time_str, market['total_turnover']/1000000)
        
        file_name = 'new_volume_file.csv'
        new = pd.read_csv(file_name)
        new.loc[len(new)] = market['total_turnover']/1000000
        new.to_csv("new_volume_file.csv", index=False)
        sum_volume = sum(new.values)
        percent2 = sum_volume[0] / 31000 * 100
        print('*** Save to file OK ***')
        print('New Turnover Sum: ', round(sum_volume[0],2))
        print('Percent: ', str(round(percent2, 2)) + '%')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


        if sum_volume[0] >= 31000:
            winsound.Beep(600, 900)

            start_all_1()


########################################################################################################



def start_all_1():

    while True:
        
        start_A()
        start_size()


########################################################################################################

### 主程序


global auto_trading
auto_trading = 1    ### 手动定日期为0，全自动为1


start_all_1()



