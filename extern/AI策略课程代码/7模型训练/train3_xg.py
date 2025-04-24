import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef
import datetime
import matplotlib.pyplot as plt
import os
import pickle

# Create temp directory if it doesn't exist
if not os.path.exists('./temp'):
    os.makedirs('./temp')

# Load the dataset
dataset_s = pd.read_csv('m99_1m_TurnoverOI_10946_6928_866.csv_tz80_Train_6032.csv')
dataset = dataset_s

num_xunlian = len(dataset_s)

# Constants
data_1_size = 866     # Test data rows
m_size = 26           # Test months
buy = 1               # Long position
sell = 0              # Short position
rrr = 0.3             # Coefficient
m = 1000              # Total capital

# Initialize result lists
res1, res2, res3, res4, res5, res6, res7 = [], [], [], [], [], [], []
resP, resR, resF = [], [], []

# Function to calculate metrics equivalent to PyCaret's metrics
def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    if y_pred_proba is None:
        y_pred_proba = y_pred  # If probabilities not provided, use predictions
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Only calculate AUC if we have probabilities and there are both classes in y_true
    auc = 0
    if len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except:
            auc = 0
    
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    results = {
        'Accuracy': accuracy,
        'AUC': auc,
        'Recall': recall,
        'Prec.': precision,
        'F1': f1,
        'Kappa': kappa,
        'MCC': mcc
    }
    
    return pd.DataFrame([results], index=['Mean'])

# Loop through PCA components
for j in range(1, 401):
    print(f"Training with {j} PCA components")
    num = j
    
    # Prepare data
    X = dataset.drop('A0', axis=1)
    y = dataset['A0']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=369)
    
    # Apply standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Apply PCA
    pca = PCA(n_components=num, random_state=369)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logitraw',
        random_state=369,
        n_jobs=-1  # Use all available CPUs
    )
    
    model.fit(X_train_pca, y_train)
    
    # Make predictions on validation set
    y_val_pred = model.predict(X_val_pca)
    y_val_pred_proba = model.predict_proba(X_val_pca)[:, 1]
    
    # Calculate and save metrics
    abc_results = calculate_metrics(y_val, y_val_pred, y_val_pred_proba)
    abc_results.to_csv('./temp/'+str(j)+'r.csv', index=False)
    
    # Save the model with scaler and PCA for later use
    model_data = {
        'model': model,
        'scaler': scaler,
        'pca': pca,
        'pca_components': num
    }
    
    with open(f'./temp/{num}x.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    # Load test data
    test_data = pd.read_csv('m99_1m_TurnoverOI_10946_6928_866.csv_tz80_Test_866_PCA.csv')
    
    # Prepare for prediction
    X_test = test_data.drop('A0', axis=1) if 'A0' in test_data.columns else test_data
    
    # Apply the same preprocessing as training
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Make predictions
    test_pred = model.predict(X_test_pca)
    test_pred_proba = model.predict_proba(X_test_pca)[:, 1]
    
    # Extract predictions for the required range
    n_preds = test_pred[0:data_1_size]
    
    # Save predictions to file
    with open(f'./temp/{num}x.txt', 'a') as Note:
        for i in range(0, data_1_size):
            Note.write(str(n_preds[i]) + '\n')
    
    # Extract prediction scores
    n_preds_score = test_pred_proba[0:data_1_size]
    
    # Save prediction scores to file
    with open(f'./temp/{num}s.txt', 'a') as Note:
        for i in range(0, data_1_size):
            Note.write(str(n_preds_score[i]) + '\n')
    
    # Update Show.csv with prediction results
    file_name = './temp/Show.csv'
    df = pd.read_csv(file_name)
    
    # Update with prediction labels
    path = f'./temp/{j}x.txt'
    df2 = pd.read_csv(path, header=None, names=['state_x'])
    for i in range(0, data_1_size):
        df['low'][i] = df2['state_x'][i]
    
    # Update with prediction scores
    path = f'./temp/{j}s.txt'
    df2 = pd.read_csv(path, header=None, names=['state_x'])
    df['score'] = 0
    for i in range(0, data_1_size):
        df['score'][i] = df2['state_x'][i]
    
    df.to_csv(f'./temp/{j}x.csv', index=False)
    
    # Load prediction results for further processing
    file_name = f'./temp/{j}x.csv'
    data_1_new = pd.read_csv(file_name)
    
    aaa1 = data_1_new['volume']
    bbb1 = data_1_new['low']
    
    # Process volume based on predictions
    if buy == 0:
        for i in range(0, data_1_size):
            if bbb1.iloc[i] == 1:
                aaa1.iloc[i] = aaa1.iloc[i] * -1
    else:
        for i in range(0, data_1_size):
            if bbb1.iloc[i] == 0:
                aaa1.iloc[i] = aaa1.iloc[i] * -1
    
    # Calculate cumulative sum
    for i in range(1, data_1_size):
        data_1_new['high'][i] = sum(data_1_new['volume'][0:(i+1)])
    
    data_1_new['high'][0] = data_1_new['volume'][0]
    
    # Calculate open values
    for i in range(0, data_1_size):
        data_1_new['open'][i] = rrr * data_1_new['high'][i] / m
    
    # Calculate win/loss statistics
    wp_win = data_1_new['volume'] > 0
    wp_lost = data_1_new['volume'] < 0
    wp_nothing = data_1_new['volume'] == 0
    
    # Count win/loss occurrences
    wp_win_a = wp_win.sum()
    wp_lost_a = wp_lost.sum()
    wp_nothing_a = wp_nothing.sum()
    
    # Calculate win/loss amounts
    rrr_win = data_1_new[wp_win]['volume'].sum()
    rrr_lost = data_1_new[wp_lost]['volume'].sum()
    
    # Calculate drawdown data
    data_1_new['down'] = 0
    log = data_1_new['open'].iloc[0]
    
    for i in range(1, len(data_1_new)):
        if data_1_new['open'].iloc[i] < log:
            data_1_new['down'].iloc[i] = data_1_new['open'].iloc[i] - log
        else:
            log = data_1_new['open'].iloc[i]
    
    # Calculate drawdown area
    downarea = sum(data_1_new['down'])
    
    # Calculate return percentage
    data_1_new['re'] = 0
    for i in range(1, len(data_1_new)):
        data_1_new['re'].iloc[i] = (data_1_new['close'].iloc[i] - data_1_new['close'].iloc[i-1]) / data_1_new['close'].iloc[i-1] * 100
    
    # Calculate actual outcomes
    data_1_new['real'] = 0
    for i in range(1, len(data_1_new)):
        if data_1_new['close'].iloc[i] < data_1_new['close'].iloc[i-1]:
            data_1_new['real'].iloc[i] = 0
        else:
            data_1_new['real'].iloc[i] = 1
    
    # Label predictions vs reality
    data_1_new['real_lab'] = 'G'
    for i in range(1, len(data_1_new)):
        if buy == 0:
            if data_1_new['low'].iloc[i] != data_1_new['real'].iloc[i]:
                data_1_new['real_lab'].iloc[i] = 'G'
            else:
                data_1_new['real_lab'].iloc[i] = 'N'
        else:
            if data_1_new['low'].iloc[i] == data_1_new['real'].iloc[i]:
                data_1_new['real_lab'].iloc[i] = 'G'
            else:
                data_1_new['real_lab'].iloc[i] = 'N'
    
    # Load show values
    file_name = './temp/Show.csv'
    df = pd.read_csv(file_name)
    data_1_new['show'] = df['low']
    
    # Label show vs predictions
    data_1_new['show_lab'] = 'G'
    for i in range(1, len(data_1_new)):
        if data_1_new['low'].iloc[i] == data_1_new['show'].iloc[i]:
            data_1_new['show_lab'].iloc[i] = 'G'
        else:
            data_1_new['show_lab'].iloc[i] = 'N'
    
    # Calculate return-based metrics
    data_1_new['re_real'] = 0
    for i in range(1, len(data_1_new)):
        if sell == 0:
            if data_1_new['low'].iloc[i] == 0:
                data_1_new['re_real'].iloc[i] = data_1_new['re'].iloc[i] * -1
            else:
                data_1_new['re_real'].iloc[i] = data_1_new['re'].iloc[i]
        else:
            if data_1_new['low'].iloc[i] == 1:
                data_1_new['re_real'].iloc[i] = data_1_new['re'].iloc[i] * -1
            else:
                data_1_new['re_real'].iloc[i] = data_1_new['re'].iloc[i]
    
    # Calculate Sharpe ratio
    sharpe = round(data_1_new['re_real'][1:].mean() / data_1_new['re_real'][1:].std() * 100, 4)
    
    # Calculate Sortino ratio
    neg_returns = data_1_new['re_real'][1:][data_1_new['re_real'][1:] < 0]
    sortino = round(data_1_new['re_real'][1:].mean() / neg_returns.std() * 100, 4) if len(neg_returns) > 0 else 0
    
    # Save processed data
    data_1_new.to_csv(f'./temp/{j}x.csv', index=False)
    
    # Calculate maximum drawdown
    s = np.argmax((np.maximum.accumulate(data_1_new['open']) - data_1_new['open']))
    if s == 0:
        e = 0
    else:
        e = np.argmax(data_1_new['open'][:s])
    
    maxdrawdown = data_1_new['open'][e] - data_1_new['open'][s]  # Maximum drawdown
    drawdown_days = s - e  # Drawdown duration
    
    # Get start and end dates for drawdown
    start_DAY = data_1_new.index[s]  # Start of drawdown
    end_DAY = data_1_new.index[e]    # End of drawdown
    start_net_value = data_1_new.iloc[s]['open']  # Start net value
    end_net_value = data_1_new.iloc[e]['open']    # End net value
    
    # Plot performance graph
    fig = plt.figure(figsize=(20, 11))
    plt.plot(data_1_new['eob'], data_1_new['open'])
    plt.plot([start_DAY, end_DAY], [start_net_value, end_net_value], linestyle='--', color='r')
    plt.xticks(range(0, data_1_size, int(data_1_size/m_size)))
    
    # Add legend with statistics
    plt.legend(['All:' + str(round(data_1_new['open'].iloc[-1]*100, 2)) + '%' +
                '   ' + str(m_size) + 'm'
                '   Year:'+ str(round(data_1_new['open'].iloc[-1]/m_size*100*12, 2)) + '%' +
                '   CalmarY:'+ str(round((data_1_new['open'].iloc[-1]/m_size*100*12)/(maxdrawdown*100), 2)) +
                '   WP:' + str(round(wp_win_a/(wp_win_a + wp_lost_a)*100, 2)) + '%' +
                '   RRR:' + str(round(rrr_win/(rrr_win+abs(rrr_lost))*100, 2)) + '%' + ' / ' + str(round(rrr_win/abs(rrr_lost), 2)) +
                '   T/N:' + str(wp_win_a + wp_lost_a ) + ' / ' + str(wp_nothing_a) +
                '   Sharpe:' + str(sharpe) +
                '   Sortino:' + str(sortino) +
                '   Accuracy:' + str(abc_results['Accuracy'][0]) +
                '   AUC:' + str(abc_results['AUC'][0]) +
                '   Recall:' + str(abc_results['Recall'][0]) +
                '   Prec:' + str(abc_results['Prec.'][0]) +
                '   F1:' + str(abc_results['F1'][0]) +
                '   Kappa:' + str(abc_results['Kappa'][0]) +
                '   MCC:' + str(abc_results['MCC'][0]),

                'MD:'+ str(round(maxdrawdown*100, 2)) + '%' +
                '   DA:'+ str(round(downarea, 4)) + '%' +
                '   MDT:' + str(drawdown_days)+
                '   Date:' + str(data_1_new['eob'].iloc[e]) + ' - ' + str(data_1_new['eob'].iloc[s])] ,

                loc='upper left',fontsize = 11)
    
    plt.plot(data_1_new['eob'], data_1_new['down'], color='#ec700a')
    plt.fill_between(data_1_new['eob'], data_1_new['down'], 0, where=(data_1_new['down']<0), facecolor='#FF0000', alpha=0.1)
    plt.xticks(range(0, data_1_size, int(data_1_size/m_size)))
    
    fig.autofmt_xdate()
    plt.grid(1)
    plt.savefig("./temp/" + str(j) + "sy.jpg")
    plt.close()

    fig = plt.figure(figsize=(20, 10))
    plt.plot(data_1_new['eob'], data_1_new['high'])
    plt.xticks(range(0, data_1_size, int(data_1_size / m_size)))
    fig.autofmt_xdate()
    plt.grid(1)
    plt.savefig("./temp/" + str(j) + "p.jpg")
    plt.close()

    # 保存评估指标结果
    pp = abc_results['Prec.'][0]
    resP.append({
        'Prec_no': j,
      'max_Prec': pp
    })

    rr = abc_results['Recall'][0]
    resR.append({
        'Recall_no': j,
      'max_Recall': rr
    })

    ff = abc_results['F1'][0]
    resF.append({
        'F1_no': j,
      'max_F1': ff
    })

    max_all = round(data_1_new['open'].iloc[-1] * 100, 2)
    max_no = j

    res1.append({
        'All_no': max_no,
      'max_All': max_all
    })

    max_CalmarY = round((data_1_new['open'].iloc[-1] / m_size * 100 * 12) / (maxdrawdown * 100), 2)
    res2.append({
        'CalmarY_no': max_no,
      'max_CalmarY': max_CalmarY
    })

    res3.append({
        'Downarea_no': max_no,
      'min_Downarea': downarea
    })

    max_wp = round(wp_win_a / (wp_win_a + wp_lost_a) * 100, 2)
    res4.append({
        'WP_no': max_no,
      'max_WP': max_wp
    })

    max_rrr = round(rrr_win / (rrr_win + abs(rrr_lost)) * 100, 2)
    res5.append({
        'RRR_no': max_no,
      'max_RRR': max_rrr
    })

    res6.append({
        'Sharpe_no': max_no,
      'max_Sharpe': sharpe
    })

    res7.append({
        'Sortino_no': max_no,
      'max_Sortino': sortino
    })

# 处理并保存 Precision、Recall 和 F1 相关结果
aaaP = pd.DataFrame(resP)
aaaR = pd.DataFrame(resR)
aaaF = pd.DataFrame(resF)

bbbP = aaaP.sort_values(by="max_Prec", ascending=False)
bbbR = aaaR.sort_values(by="max_Recall", ascending=False)
bbbF = aaaF.sort_values(by="max_F1", ascending=False)

bbbP = bbbP.reset_index(drop=True)
bbbR = bbbR.reset_index(drop=True)
bbbF = bbbF.reset_index(drop=True)

bbbP['Recall_no'] = bbbR['Recall_no']
bbbP['max_Recall'] = bbbR['max_Recall']
bbbP['F1_no'] = bbbF['F1_no']
bbbP['max_F1'] = bbbF['max_F1']

bbbP.to_csv("./temp/Best_2.csv", index=False)

# 处理并保存其他评估指标结果
aaa1 = pd.DataFrame(res1)
aaa2 = pd.DataFrame(res2)
aaa3 = pd.DataFrame(res3)
aaa4 = pd.DataFrame(res4)
aaa5 = pd.DataFrame(res5)
aaa6 = pd.DataFrame(res6)
aaa7 = pd.DataFrame(res7)

bbb1 = aaa1.sort_values(by="max_All", ascending=False)
bbb2 = aaa2.sort_values(by="max_CalmarY", ascending=False)
bbb3 = aaa3.sort_values(by="min_Downarea", ascending=False)
bbb4 = aaa4.sort_values(by="max_WP", ascending=False)
bbb5 = aaa5.sort_values(by="max_RRR", ascending=False)
bbb6 = aaa6.sort_values(by="max_Sharpe", ascending=False)
bbb7 = aaa7.sort_values(by="max_Sortino", ascending=False)

bbb1 = bbb1.reset_index(drop=True)
bbb2 = bbb2.reset_index(drop=True)
bbb3 = bbb3.reset_index(drop=True)
bbb4 = bbb4.reset_index(drop=True)
bbb5 = bbb5.reset_index(drop=True)
bbb6 = bbb6.reset_index(drop=True)
bbb7 = bbb7.reset_index(drop=True)

bbb1['CalmarY_no'] = bbb2['CalmarY_no']
bbb1['max_CalmarY'] = bbb2['max_CalmarY']
bbb1['Downarea_no'] = bbb3['Downarea_no']
bbb1['min_Downarea'] = bbb3['min_Downarea']
bbb1['WP_no'] = bbb4['WP_no']
bbb1['max_WP'] = bbb4['max_WP']
bbb1['RRR_no'] = bbb5['RRR_no']
bbb1['max_RRR'] = bbb5['max_RRR']
bbb1['Sharpe_no'] = bbb6['Sharpe_no']
bbb1['max_Sharpe'] = bbb6['max_Sharpe']
bbb1['Sortino_no'] = bbb7['Sortino_no']
bbb1['max_Sortino'] = bbb7['max_Sortino']

bbb1.to_csv("./temp/Best_1.csv", index=False)