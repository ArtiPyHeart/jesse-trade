from tqsdk import TqAccount, TqApi, TqAuth, TargetPosTask
import time
import winsound


api = TqApi(TqAccount("期货公司", "账号", "密码"), auth=TqAuth("天勤账号", "天勤密码"))

target_pos_task_rb = TargetPosTask(api, "SHFE.rb2505")   ### 交易的合约

with open('RB_A.txt', 'r') as file:
    content = file.read()

c_rb = int(content)

if c_rb == 0: c_rb = -1
if c_rb == 2: c_rb = 0


target_pos_task_rb.set_target_volume(c_rb*20)  ### 交易手数


abc = 1

while True:
    api.wait_update()

    abc = abc + 1    
    print(abc, c_rb)
    print('-------------------------------')
    
    with open('RB_A.txt', 'r') as file:
        content_new = file.read()

    try:
        n_rb = int(content_new)

        if n_rb == 0: n_rb = -1
        if n_rb == 2: n_rb = 0

    except Exception as e:
        print(f"Error: {e}")
        winsound.Beep(600,900)
        continue

    if n_rb != c_rb:
        c_rb = n_rb
        
        target_pos_task_rb.set_target_volume(c_rb*20)   ### 交易手数


    time.sleep(1)










