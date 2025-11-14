# %%
import pandas as pd
import numpy as np
from collections import defaultdict
import joblib
import torch
import torch.nn.functional as F
from model import IFS_Former 
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader 
import time
import os
import random
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import glob
import copy

# %%
num_epochs = 10
batch_size = 32 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_target = 1
n_cat = 24
n_nume = 72

max_min_dict = {'连续特征0_行程距离': [369.5407184179599, 0.9500099962155564], '连续特征1_近期平均能耗率': [0.5988776236111111, 0.0064202106509405], '连续特征2_累计平均能耗率': [0.34904404305555553, 0.0711048941274899], '连续特征3_近期能量回收比例': [0.7948677720321674, 2.2960322678524668e-05], '连续特征4_近期加速踏板均值': [54.86303030303031, 5.018333333333334], '连续特征5_近期加速踏板最大值': [99.66666666666669, 1.6666666666666667], '连续特征6_近期加速踏板标准差': [36.2602439260524, 0.5592022758762369], '连续特征7_近期制动踏板均值': [39.99120473022912, 3.1559983896940444], '连续特征8_近期制动踏板最大值': [90.66666666666669, 5.0], '连续特征9_近期制动踏板标准差': [22.187466186883952, 0.3043903499227176], '连续特征10_近期瞬时速度均值': [111.12637789395552, 1.5476702376242475], '连续特征11_近期瞬时速度最大值': [164.66666666666666, 21.0], '连续特征12_近期瞬时速度标准差': [47.34461568746429, 5.4295127125395934], '连续特征13_近期加速度均值': [0.6016172800877638, 0.03723403011597455], '连续特征14_近期加速度最大值': [2.7620370370370373, 0.09259259259259259], '连续特征15_近期加速度标准差': [0.5414802313409247, 0.033274441993689306], '连续特征16_近期减速度均值': [-0.03989025375611458, -0.6078574556066182], '连续特征17_近期减速度最大值': [-0.08950617283950613, -2.75925925925926], '连续特征18_近期减速度标准差': [0.5390875984724751, 0.03137910647036119], '连续特征19_当前SOC': [100.0, 5.0], '连续特征20_当前累积行驶里程': [540036.0, 0.0], '连续特征21_当前单体电池电压极差': [4.094, 0.0], '连续特征22_当前单体电池温度极差': [78.0, 0.0], '连续特征23_当前绝缘电阻值': [60000.0, 0.0], '连续特征24_平均充电时长': [1137.5333333333333, 10.683333333333334], '连续特征25_最大充电时长': [1434.1833333333334, 10.683333333333334], '连续特征26_最小充电时长': [1137.5333333333333, 10.0], '连续特征27_起始SOC均值': [95.5, 5.0], '连续特征28_截止SOC均值': [100.0, 15.0], '连续特征29_充电SOC均值': [95.0, 0.0], '连续特征30_温度': [39.2, -10.2], '连续特征31_气压mmHg': [782.2, 695.4], '连续特征32_相对湿度': [100.0, 10.0], '连续特征33_风速m/s': [11.0, 0.0], '连续特征34_能见度km': [30.0, 0.1], '连续特征35_降水量mm': [98.0, 0.0], '连续特征36_满载质量': [3100.0, 1900.0], '连续特征37_整备质量': [2300.0, 1400.0], '连续特征38_电池额定能量': [82.8, 35.0], '连续特征39_电池额定容量': [216.2, 100.0], '连续特征40_最大功率': [363.0, 80.0], '连续特征41_最大扭矩': [680.0, 130.0], '连续特征42_官方百公里能耗': [17.0, 12.2], '连续特征43_行程时间': ['nan', 'nan'], '连续特征44_行程平均速度(d/t)': ['nan', 'nan'], '连续特征45': ['nan', 'nan'], '连续特征46': ['nan', 'nan'], '连续特征47': ['nan', 'nan'], '连续特征48': ['nan', 'nan'], '连续特征49': ['nan', 'nan'], '连续特征50': ['nan', 'nan'], '连续特征51': ['nan', 'nan'], '连续特征52': ['nan', 'nan'], '连续特征53': ['nan', 'nan'], '连续特征54': ['nan', 'nan'], '连续特征55': ['nan', 'nan'], '连续特征56': ['nan', 'nan'], '连续特征57': ['nan', 'nan'], '连续特征58': ['nan', 'nan'], '连续特征59': ['nan', 'nan'], '连续特征60': ['nan', 'nan'], '连续特征61': ['nan', 'nan'], '连续特征62': ['nan', 'nan'], '连续特征63': ['nan', 'nan'], '连续特征64': ['nan', 'nan'], '连续特征65': ['nan', 'nan'], '连续特征66': ['nan', 'nan'], '连续特征67': ['nan', 'nan'], '连续特征68': ['nan', 'nan'], '连续特征69': ['nan', 'nan'], '连续特征70': ['nan', 'nan'], '连续特征71': ['nan', 'nan'], '行程能耗': [51.98927375, 0.1000901111111111]}

# %%
# 自定义数据集类

class AMFormerTensorDataset(Dataset):
    def __init__(self, df, nume_features, cat_features, targets,
                 mode=None,
                 droppable_feature_groups=None,
                 dropout_probs=None,
                 cat_mask_value = -5, # 表征离散变量的缺失值
                 nume_mask_value = np.nan):
        
        self.mode = mode
        self.droppable_feature_groups = droppable_feature_groups or {}
        self.dropout_probs = dropout_probs or {}
        self.cat_mask_value = cat_mask_value
        self.nume_mask_value = nume_mask_value

        self.nume_features_tensor = torch.tensor(
            df[nume_features].values, 
            dtype=torch.float32
        )
        self.cat_features_tensor = torch.tensor(
            df[cat_features].values, 
            dtype=torch.long
        )
        self.targets_tensor = torch.tensor(
            df[targets].values, 
            dtype=torch.float32
        )

        self.nume_feature_names = nume_features
        self.cat_feature_names = cat_features

    def __len__(self):
        return len(self.targets_tensor)

    def __getitem__(self, idx):
        x_nume = self.nume_features_tensor[idx].clone()
        x_cat = self.cat_features_tensor[idx].clone()
        target = self.targets_tensor[idx].clone()

        return {
            'x_cat': x_cat,
            'x_nume': x_nume,
            'target': target
        }

# %%
feature_groups = {
    # 必须保留的特征
    '基础特征': ['连续特征0_行程距离', '连续特征1_近期平均能耗率', '连续特征2_累计平均能耗率', '离散特征2_当前月份', '离散特征3_当前几点', '离散特征4_当前星期几'],

    # 可去除的特征
    '驾驶行为特征': ['连续特征3_近期能量回收比例', '连续特征4_近期加速踏板均值', '连续特征5_近期加速踏板最大值',
                '连续特征6_近期加速踏板标准差', '连续特征7_近期制动踏板均值', '连续特征8_近期制动踏板最大值',
                '连续特征9_近期制动踏板标准差', '连续特征10_近期瞬时速度均值', '连续特征11_近期瞬时速度最大值',
                '连续特征12_近期瞬时速度标准差', '连续特征13_近期加速度均值', '连续特征14_近期加速度最大值',
                '连续特征15_近期加速度标准差', '连续特征16_近期减速度均值', '连续特征17_近期减速度最大值',
                '连续特征18_近期减速度标准差'],
    '充电行为特征': ['连续特征19_当前SOC', '连续特征20_当前累积行驶里程',
                '连续特征21_当前单体电池电压极差', '连续特征22_当前单体电池温度极差', '连续特征23_当前绝缘电阻值',
                '连续特征24_平均充电时长', '连续特征25_最大充电时长', '连续特征26_最小充电时长', '连续特征27_起始SOC均值',
                '连续特征28_截止SOC均值', '连续特征29_充电SOC均值'],
    '气象特征': ['连续特征30_温度', '连续特征31_气压mmHg',
                '连续特征32_相对湿度', '连续特征33_风速m/s', '连续特征34_能见度km', '连续特征35_降水量mm'],
    '静态特征': ['离散特征0_车型', '离散特征1_电池类型', '连续特征36_满载质量', '连续特征37_整备质量', '连续特征38_电池额定能量', '连续特征39_电池额定容量',
                '连续特征40_最大功率', '连续特征41_最大扭矩', '连续特征42_官方百公里能耗'],

    # 新增特征
    '交通特征': ['连续特征43_行程时间', '连续特征44_行程平均速度(d/t)'],
}

# %%
file_ls = {
    '0.csv': ['基础特征', '充电行为特征', '气象特征', '静态特征'], # 无驾驶
    '1.csv': ['基础特征', '驾驶行为特征', '气象特征', '静态特征'], # 无充电
    '2.csv': ['基础特征', '驾驶行为特征', '充电行为特征', '静态特征'], # 无气象
    '3.csv': ['基础特征', '驾驶行为特征', '充电行为特征', '气象特征'], # 无静态
    '4.csv': ['基础特征', '充电行为特征', '气象特征', '静态特征','交通特征'], # 无驾驶 加交通
    '5.csv': ['基础特征', '驾驶行为特征', '充电行为特征', '气象特征', '静态特征'], # 原始特征
}

# %%
save_dir = './results/downstream/IFS_former/few_shot'
few_shot_ls = [32, 64, 128, 256, 384]

for few in few_shot_ls:
    print('\n',few)
    os.makedirs(f'{save_dir}/{few}', exist_ok=True) 

    result_dict = defaultdict(list) 
    all_max_min = dict()

    total_train_time = 0

    for file in file_ls:
        dfdata = pd.read_csv(f'./dataset/test/{file}', encoding='utf-8')
        print(file)

        # 提取需要用到的列
        all_features = []
        for feature_type in file_ls[file]:
            all_features.extend(feature_groups[feature_type])

        target_cols = dfdata.columns[0:1].to_list()
        cat_cols = dfdata.columns[1:25].to_list()
        nume_cols = dfdata.columns[25:97].to_list()

        dfdata_train_raw = dfdata.iloc[:few].reset_index(drop=True)  # 训练集few_shot_ls对齐
        dfdata_test_raw = dfdata.iloc[384:].reset_index(drop=True)  # 测试集对齐

        cat_cols_use = [cat for cat in cat_cols if cat in all_features] # 其余列将直接填为缺失，但mask只有没开启的特征位
        nume_cols_use = [nume for nume in nume_cols if nume in all_features] # 其余列将直接填为缺失，但mask只有没开启的特征位

        dfdata_train = dfdata_train_raw.copy(deep=True)
        dfdata_test = dfdata_test_raw.copy(deep=True)
        max_min_dict_save = copy.deepcopy(max_min_dict)

        # 特征归一化
        for name in nume_cols_use:
            v_max = max_min_dict_save[name][0]
            v_min = max_min_dict_save[name][1]
            if v_max != 'nan' and v_min != 'nan':
                dfdata_train[name] = (dfdata_train_raw[name] - v_min) / (v_max - v_min)
                dfdata_test[name] = (dfdata_test_raw[name] - v_min) / (v_max - v_min)
            else:
                if len(name.split('_'))>1:
                    # 如果是行程时间和行程平均速度(d/t)
                    v_max = dfdata_train[name].max()
                    v_min = dfdata_train[name].min()
                    dfdata_train[name] = (dfdata_train_raw[name] - v_min) / (v_max - v_min)
                    dfdata_test[name] = (dfdata_test_raw[name] - v_min) / (v_max - v_min)

                    max_min_dict_save[name] = [v_max, v_min] # 更新用于保存的dict

        # 目标归一化
        target_name = target_cols[0]
        target_max = max_min_dict_save[target_name][0]
        target_min = max_min_dict_save[target_name][1]
        dfdata_train[target_name] = (dfdata_train_raw[target_name] - target_min) / (target_max - target_min)
        dfdata_test[target_name] = (dfdata_test_raw[target_name] - target_min) / (target_max - target_min)
        
        all_max_min[file] = max_min_dict_save

        # del dfdata 

        # 将一些列填充为缺失       
        for name in cat_cols:
            if name not in cat_cols_use:
                dfdata_train[name] = -5 # 将未使用的离散特征填充为-5，表示缺失值
                dfdata_test[name] = -5 # 将未使用的离散特征填充为-5，表示缺失值
        for name in nume_cols:
            if name not in nume_cols_use:
                dfdata_train[name] = np.nan # 将未使用的连续特征填充为np.nan，表示缺失值
                dfdata_test[name] = np.nan # 将未使用的连续特征填充为np.nan，表示缺失值

        train_dataset = AMFormerTensorDataset(
            df=dfdata_train,
            nume_features=nume_cols,
            cat_features=cat_cols,
            targets=target_cols,
            mode='fine-tune'
        )

        test_dataset = AMFormerTensorDataset(
            df=dfdata_test,
            nume_features=nume_cols,
            cat_features=cat_cols,
            targets=target_cols,
            mode='test'
        )

        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True, 
            pin_memory=True, 
            drop_last=False 
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True, 
            drop_last=False 
        )

        # mask需要根据实际情况动态调整
        if file in ['4.csv']: # 只有这两辆车加了交通特征
            mask_positions = [_ for _ in range(5,24)] + [_ for _ in range(24+45,96)] 
        else:
            mask_positions = [_ for _ in range(5,24)] + [_ for _ in range(24+43,96)]

        model = IFS_Former(
            categories = [32]*n_cat,     
            num_continuous = n_nume,             
            dim = 256,                         
            dim_out = n_target,                   
            depth=6,                             
            heads=8,                             
            attn_dropout=0.1,                    
            ff_dropout=0.1,                      
            groups=[96]*6,                    
            sum_num_per_group=[16]*6,        
            prod_num_per_group=[16]*6,         
            mask_specific_columns=mask_positions 
        )
        model.load_state_dict(torch.load('./checkpoint/IFS_Former_checkpoint.pth', weights_only=True, map_location=device))
        model.to(device)


        new_params_group_list = [
            model.numerical_embedder.weights,
            model.numerical_embedder.biases,
            model.missing_feature_embedding 
        ]

        grouped_ids = set(id(p) for p in new_params_group_list)

        pretrained_parts_group_list = []
        for p in model.parameters():
            if id(p) not in grouped_ids:
                pretrained_parts_group_list.append(p)


        optimizer = torch.optim.AdamW([
            {'params': new_params_group_list, 'lr': 5e-6, 'name': 'new_related_features'},
            {'params': pretrained_parts_group_list, 'lr': 0.0, 'name': 'pretrained_parts'}
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)

        t1 = time.perf_counter()
        for epoch in range(num_epochs):
        
            model.train()
            train_loss = 0
            train_loss_inverse_transform = 0

            train_actuals = []
            train_preds = []

            for batch in train_loader:
                
                x_cat = batch['x_cat'].long().to(device)      
                x_nume = batch['x_nume'].float().to(device)   
                y = batch['target'].float().squeeze().to(device)        

                optimizer.zero_grad()
                pred = model(x_cat, x_nume).squeeze() # 去掉是1的维度
                loss = F.mse_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2) # 梯度裁剪
                optimizer.step()
                train_loss += loss.item()

                loss_inverse_transform = F.mse_loss(
                    pred * (target_max - target_min) + target_min, 
                    y * (target_max - target_min) + target_min
                )
                train_loss_inverse_transform += loss_inverse_transform.item()

                train_actuals.extend(y.cpu().numpy().tolist())
                train_preds.extend(pred.cpu().detach().numpy().tolist())

            avg_train_loss = train_loss / len(train_loader)
            avg_train_loss_inverse_transform = train_loss_inverse_transform / len(train_loader)
            train_r2 = r2_score(train_actuals, train_preds)

            scheduler.step() 

            if epoch == 5-1:
                optimizer.param_groups[0]['lr'] = 5e-7 
                optimizer.param_groups[1]['lr'] = 5e-7
                
        t2 = time.perf_counter()
        total_train_time+=(t2-t1)

        torch.save(model.state_dict(), f'{save_dir}/{few}/{file}_model.pth') 

        # 下游测试集
        model.eval()

        train_actuals = []
        train_preds = []

        test_actuals = []
        test_preds = []

        with torch.no_grad():
            for batch in train_loader:
                x_cat = batch['x_cat'].long().to(device)      
                x_nume = batch['x_nume'].float().to(device)   
                y = batch['target'].float().squeeze().to(device)  
                
                pred = model(x_cat, x_nume).squeeze() 
                pred = pred * (target_max - target_min) + target_min
                y = y * (target_max - target_min) + target_min

                train_actuals.extend(y.cpu().numpy().tolist())
                train_preds.extend(pred.cpu().numpy().tolist())  

            for batch in test_loader:
                x_cat = batch['x_cat'].long().to(device)      
                x_nume = batch['x_nume'].float().to(device)   
                y = batch['target'].float().squeeze().to(device)  
                
                pred = model(x_cat, x_nume).squeeze()
                pred = pred * (target_max - target_min) + target_min
                y = y * (target_max - target_min) + target_min

                test_actuals.extend(y.cpu().numpy().tolist())
                test_preds.extend(pred.cpu().numpy().tolist()) 
        
        r2_train = r2_score(train_actuals, train_preds)
        mae_train = mean_absolute_error(train_actuals, train_preds)
        rmse_train = mean_squared_error(train_actuals, train_preds)**0.5
        mape_train = mean_absolute_percentage_error(train_actuals, train_preds)

        r2_test = r2_score(test_actuals, test_preds)
        mae_test = mean_absolute_error(test_actuals, test_preds)
        rmse_test = mean_squared_error(test_actuals, test_preds)**0.5
        mape_test = mean_absolute_percentage_error(test_actuals, test_preds)

        # 保存结果
        result_dict['file'].append(file)  

        result_dict['r2_train'].append(r2_train)
        result_dict['mae_train'].append(mae_train)
        result_dict['rmse_train'].append(rmse_train)
        result_dict['mape_train'].append(mape_train)
        result_dict['y_mean_train'].append(np.array(dfdata_train_raw['行程能耗']).mean())

        result_dict['r2_test'].append(r2_test)
        result_dict['mae_test'].append(mae_test)
        result_dict['rmse_test'].append(rmse_test)
        result_dict['mape_test'].append(mape_test)
        result_dict['y_mean_test'].append(np.array(dfdata_test_raw['行程能耗']).mean())  

        print(f"训练集 R2: {round(r2_train, 6)}", f"MAE: {round(mae_train, 6)}", f"RMSE: {round(rmse_train, 6)}", f"MAPE: {round(mape_train, 6)}")
        print(f"测试集 R2: {round(r2_test, 6)}", f"MAE: {round(mae_test, 6)}", f"RMSE: {round(rmse_test, 6)}", f"MAPE: {round(mape_test, 6)}")
   
    joblib.dump(all_max_min, f'{save_dir}/{few}/max_min.joblib')

    # DataFrame
    results_df = pd.DataFrame({
        'File': result_dict['file'],
        'R2_Train': result_dict['r2_train'],
        'MAE_Train': result_dict['mae_train'],
        'RMSE_Train': result_dict['rmse_train'],
        'MAPE_Train': result_dict['mape_train'],
        'Y_Mean_Train': result_dict['y_mean_train'], 
        'R2_Test': result_dict['r2_test'],
        'MAE_Test': result_dict['mae_test'],
        'RMSE_Test': result_dict['rmse_test'],
        'MAPE_Test': result_dict['mape_test'],
        'Y_Mean_Test': result_dict['y_mean_test']
    })

    # 
    summary_row = pd.DataFrame({
        'File': ['SUMMARY'],
        'R2_Train': [f"{np.mean(result_dict['r2_train'])} ± {np.std(result_dict['r2_train'])}"],
        'MAE_Train': [f"{np.mean(result_dict['mae_train'])} ± {np.std(result_dict['mae_train'])}"],
        'RMSE_Train': [f"{np.mean(result_dict['rmse_train'])} ± {np.std(result_dict['rmse_train'])}"],
        'MAPE_Train': [f"{np.mean(result_dict['mape_train'])} ± {np.std(result_dict['mape_train'])}"],
        'Y_Mean_Train': ["vacant"],
        'R2_Test': [f"{np.mean(result_dict['r2_test'])} ± {np.std(result_dict['r2_test'])}"],
        'MAE_Test': [f"{np.mean(result_dict['mae_test'])} ± {np.std(result_dict['mae_test'])}"],
        'RMSE_Test': [f"{np.mean(result_dict['rmse_test'])} ± {np.std(result_dict['rmse_test'])}"],
        'MAPE_Test': [f"{np.mean(result_dict['mape_test'])} ± {np.std(result_dict['mape_test'])}"],
        'Y_Mean_Test': ["vacant"]
    })

    results_df = pd.concat([results_df, summary_row], ignore_index=True)

    print(f"AMF R2: {np.mean(result_dict['r2_test'])} ± {np.std(result_dict['r2_test'])}")
    print(f"AMF MAE: {np.mean(result_dict['mae_test'])} ± {np.std(result_dict['mae_test'])}")
    print(f"AMF RMSE: {np.mean(result_dict['rmse_test'])} ± {np.std(result_dict['rmse_test'])}")
    print(f"AMF MAPE: {np.mean(result_dict['mape_test'])} ± {np.std(result_dict['mape_test'])}")

    results_df.to_csv(f'{save_dir}/{few}/results.csv', index=False, encoding='utf-8-sig')  
    print('\n file saved \n')

    print('总训练时间(秒): ', total_train_time)
    print('平均训练时间(秒): ', total_train_time/13)
    

    print('\n')
