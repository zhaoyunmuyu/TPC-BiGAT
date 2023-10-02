'''
 Filename:  train.py
 Description:  模型训练过程
 Created:  2022年11月2日 16时49分
 Author:  Li Shao
'''

import os
import time
import transformers
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from config.config import Config
from process.dataset import *
from model.BiGCN import BiGCN
import torch.nn.functional as F
from utils.evaluate import Evaluation4Class, Evaluation2Class
from utils.dataset_split import dataset_split
from utils.earlystopping import EarlyStopping2Class, EarlyStopping4Class
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import lr_scheduler


def train_model_2class(config,model,x_train,x_test,iter):
    config.logger.info("Start:生成Dataloader")
    traindata_list,testdata_list = loadBiData(x_train,x_test,config.result_graph_dir,config.logger,config.droprate)
    train_loader = DataLoader(traindata_list, batch_size=config.batch_size, shuffle=True, num_workers=config.number_workers)
    test_loader = DataLoader(testdata_list, batch_size=config.batch_size, shuffle=True, num_workers=config.number_workers)
    config.logger.info("Start:初始化模型")
    model_save_path = os.path.join(config.cache_save_dir, model.get_name()+'_'+config.dataset+'.pkl')
    # if os.path.exists(model_save_path):
    #     loaded_paras = torch.load(model_save_path)
    #     model.load_state_dict(loaded_paras)
    #     config.logger.info("##成功载入已有模型，进行追加训练##")
    model = model.to(config.device)
    optimizer = transformers.AdamW([{'params': model.parameters()}],
                                    lr=config.learning_rate,
                                    weight_decay=config.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.002)
    early_stopping = EarlyStopping2Class(patience=config.patience,verbose=True) 
    config.logger.info(config.log_config(model.get_name()))
    config.logger.info("Start:开始训练")
    model.train()
    train_losses,val_losses,train_accses,val_accs = [],[],[],[]
    for epoch in range(config.epochs):
        batch_idx = 0
        start_time = time.time()
        avg_loss,avg_acc = [],[]
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            x_data, y_label = Batch_data[0], Batch_data[1]
            x_data = x_data.to(config.device)
            y_label = y_label.to(config.device)
            out_labels= model(x_data)
            loss = F.nll_loss(out_labels, y_label.y)
            # crossloss = torch.nn.CrossEntropyLoss()
            # loss = crossloss(out_labels, y_label.y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2) # 梯度剪裁
            avg_loss.append(loss.item())
            optimizer.step()
            # result
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(y_label.y).sum().item()
            train_acc = correct / len(y_label.y)
            avg_acc.append(train_acc)
            # print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,loss.item(),train_acc))
            batch_idx += 1
        # scheduler.step()
        end_time = time.time()
        train_loss = np.mean(avg_loss)
        train_accs = np.mean(avg_acc)
        train_losses.append(train_loss)
        train_accses.append(train_accs)
        msg = f"Iter: {iter},Epoch: {epoch},Tra-loss: {train_loss:.4f},Tra-Acc: {train_accs:.4f},Epoch time: {(end_time - start_time):.3f}s"
        config.logger.info(msg)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), model_save_path)
        # evaluate
        val_avg_loss, val_avg_acc = [],[]
        val_Acc_all = []
        val_Acc1, val_Pre1, val_Rec1, val_F1_1 = [], [], [], []
        val_Acc2, val_Pre2, val_Rec2, val_F1_2 = [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            x_data, y_label = Batch_data[0], Batch_data[1]
            x_data = x_data.to(config.device)
            y_label = y_label.to(config.device)
            val_out= model(x_data)
            val_loss = F.nll_loss(val_out, y_label.y)
            val_avg_loss.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(y_label.y).sum().item()
            val_acc = correct / len(y_label.y)
            val_avg_acc.append(val_acc)
            # 详细指标
            Acc_all, Acc1, Pre1, Rec1, F1_1, Acc2, Pre2, Rec2, F1_2 = Evaluation2Class(val_pred, y_label.y)
            val_Acc_all.append(Acc_all)
            val_Acc1.append(Acc1),val_Pre1.append(Pre1),val_Rec1.append(Rec1),val_F1_1.append(F1_1)
            val_Acc2.append(Acc2),val_Pre2.append(Pre2),val_Rec2.append(Rec2),val_F1_2.append(F1_2)
        val_losses.append(np.mean(val_avg_loss))
        val_accs.append(np.mean(val_avg_acc))
        msg = f"Iter: {iter},Epoch: {epoch},Val-loss: {np.mean(val_avg_loss):.4f},Val-Acc: {np.mean(val_avg_acc):.4f}"
        config.logger.info(msg)
        result = {}
        result['accs'] = np.mean(val_Acc_all)
        result['acc1'] = np.mean(val_Acc1)
        result['acc2'] = np.mean(val_Acc2)
        result['pre1'] = np.mean(val_Pre1)
        result['pre2'] = np.mean(val_Pre2)
        result['rec1'] = np.mean(val_Rec1)
        result['rec2'] = np.mean(val_Rec2)
        result['F1_1'] = np.mean(val_F1_1)
        result['F1_2'] = np.mean(val_F1_2)
        msg = f"Iter: {iter},Epoch: {epoch}\n\
                Acc:{result['accs']:.3f};\n\
                Acc1:{result['acc1']:.3f},Pre1:{result['pre1']:.3f},Rec1:{result['rec1']:.3f},F1_1:{result['F1_1']:.3f};\n\
                Acc2:{result['acc2']:.3f},Pre2:{result['pre2']:.3f},Rec2:{result['rec2']:.3f},F1_2:{result['F1_2']:.3f};"
        config.logger.info(msg)
        early_stopping(np.mean(val_avg_loss), np.mean(val_avg_acc), np.mean(val_F1_1), np.mean(val_F1_2), model, model_save_path)
        accs =np.mean(val_avg_acc)
        F1_1 = np.mean(val_F1_1)
        F1_2 = np.mean(val_F1_2)
        if early_stopping.early_stop:
            config.logger.info("Early stopping")
            accs = early_stopping.accs
            F1_1 = early_stopping.F1_1
            F1_2 = early_stopping.F1_2
            break
    result = f"train_losses: {train_losses};\n\
               val_losses: {val_losses};\n\
               train_accses: {train_accses};\n\
               val_accs: {val_accs};\n\
               accs:{accs},F1_1:{F1_1},F1_2:{F1_2};"
    config.logger.info(result)
    return train_losses,val_losses,train_accses,val_accs,accs,F1_1,F1_2

def train_model_4class(config,model,x_train,x_test,iter):
    config.logger.info("Start:生成Dataloader")
    traindata_list,testdata_list = loadBiData(x_train,x_test,config.result_graph_dir,config.logger,config.droprate)
    train_loader = DataLoader(traindata_list, batch_size=config.batch_size, shuffle=True, num_workers=config.number_workers)
    test_loader = DataLoader(testdata_list, batch_size=config.batch_size, shuffle=True, num_workers=config.number_workers)
    config.logger.info("Start:初始化模型")
    model_save_path = os.path.join(config.cache_save_dir, model.get_name()+'_'+config.dataset+'.pkl')
    # if os.path.exists(model_save_path):
    #     loaded_paras = torch.load(model_save_path)
    #     model.load_state_dict(loaded_paras)
    #     config.logger.info("##成功载入已有模型，进行追加训练##")
    model = model.to(config.device)
    optimizer = transformers.AdamW([{'params': model.parameters()}],
                                    lr=config.learning_rate,
                                    weight_decay=config.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.002)
    early_stopping = EarlyStopping4Class(patience=config.patience,verbose=True) 
    config.logger.info(config.log_config(model.get_name()))
    config.logger.info("Start:开始训练")
    model.train()
    train_losses,val_losses,train_accses,val_accs = [],[],[],[]
    for epoch in range(config.epochs):
        batch_idx = 0
        start_time = time.time()
        avg_loss,avg_acc = [],[]
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            x_data, y_label = Batch_data[0], Batch_data[1]
            x_data = x_data.to(config.device)
            y_label = y_label.to(config.device)
            out_labels= model(x_data)
            loss = F.nll_loss(out_labels, y_label.y)
            # crossloss = torch.nn.CrossEntropyLoss()
            # loss = crossloss(out_labels, y_label.y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2) # 梯度剪裁
            avg_loss.append(loss.item())
            optimizer.step()
            # result
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(y_label.y).sum().item()
            train_acc = correct / len(y_label.y)
            avg_acc.append(train_acc)
            # print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,loss.item(),train_acc))
            batch_idx += 1
        # scheduler.step()
        end_time = time.time()
        train_loss = np.mean(avg_loss)
        train_accs = np.mean(avg_acc)
        train_losses.append(train_loss)
        train_accses.append(train_accs)
        msg = f"Iter: {iter},Epoch: {epoch},Tra-loss: {train_loss:.4f},Tra-Acc: {train_accs:.4f},Epoch time: {(end_time - start_time):.3f}s"
        config.logger.info(msg)
        if epoch % 5 == 0:
            torch.save(model.state_dict(), model_save_path)
        # evaluate
        val_Acc_all = []
        val_Acc1, val_Pre1, val_Rec1, val_F1_1 = [], [], [], []
        val_Acc2, val_Pre2, val_Rec2, val_F1_2 = [], [], [], []
        val_Acc3, val_Pre3, val_Rec3, val_F1_3 = [], [], [], []
        val_Acc4, val_Pre4, val_Rec4, val_F1_4 = [], [], [], []
        val_avg_loss, val_avg_acc = [],[]
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            x_data, y_label = Batch_data[0], Batch_data[1]
            x_data = x_data.to(config.device)
            y_label = y_label.to(config.device)
            val_out= model(x_data)
            val_loss = F.nll_loss(val_out, y_label.y)
            val_avg_loss.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(y_label.y).sum().item()
            val_acc = correct / len(y_label.y)
            val_avg_acc.append(val_acc)
            # 详细指标
            Acc_all, Acc1, Pre1, Rec1, F1_1, Acc2, Pre2, Rec2, F1_2, Acc3, Pre3, Rec3, F1_3, Acc4, Pre4, Rec4, F1_4 = Evaluation4Class(val_pred, y_label.y)
            val_Acc_all.append(Acc_all)
            val_Acc1.append(Acc1),val_Pre1.append(Pre1),val_Rec1.append(Rec1),val_F1_1.append(F1_1)
            val_Acc2.append(Acc2),val_Pre2.append(Pre2),val_Rec2.append(Rec2),val_F1_2.append(F1_2)
            val_Acc3.append(Acc3),val_Pre3.append(Pre3),val_Rec3.append(Rec3),val_F1_3.append(F1_3)
            val_Acc4.append(Acc4),val_Pre4.append(Pre4),val_Rec4.append(Rec4),val_F1_4.append(F1_4)
        val_losses.append(np.mean(val_avg_loss))
        val_accs.append(np.mean(val_avg_acc))
        msg = f"Iter: {iter},Epoch: {epoch},Val-loss: {np.mean(val_avg_loss):.4f},Val-Acc: {np.mean(val_avg_acc):.4f}"
        config.logger.info(msg)
        result = {}
        result['accs'] = np.mean(val_Acc_all)
        result['acc1'] = np.mean(val_Acc1)
        result['acc2'] = np.mean(val_Acc2)
        result['acc3'] = np.mean(val_Acc3)
        result['acc4'] = np.mean(val_Acc4)
        result['pre1'] = np.mean(val_Pre1)
        result['pre2'] = np.mean(val_Pre2)
        result['pre3'] = np.mean(val_Pre3)
        result['pre4'] = np.mean(val_Pre4)
        result['rec1'] = np.mean(val_Rec1)
        result['rec2'] = np.mean(val_Rec2)
        result['rec3'] = np.mean(val_Rec3)
        result['rec4'] = np.mean(val_Rec4)
        result['F1_1'] = np.mean(val_F1_1)
        result['F1_2'] = np.mean(val_F1_2)
        result['F1_3'] = np.mean(val_F1_3)
        result['F1_4'] = np.mean(val_F1_4)
        msg = f"Iter: {iter},Epoch: {epoch}\n\
                Acc:{result['accs']:.3f};\n\
                Acc1:{result['acc1']},Pre1:{result['pre1']},Rec1:{result['rec1']},F1_1:{result['F1_1']};\n\
                Acc2:{result['acc2']},Pre2:{result['pre2']},Rec2:{result['rec2']},F1_2:{result['F1_2']};\n\
                Acc3:{result['acc3']},Pre3:{result['pre3']},Rec3:{result['rec3']},F1_3:{result['F1_3']};\n\
                Acc4:{result['acc4']},Pre4:{result['pre4']},Rec4:{result['rec4']},F1_4:{result['F1_4']};"
        config.logger.info(msg)
        early_stopping(np.mean(val_avg_loss), np.mean(val_avg_acc), np.mean(val_F1_1), np.mean(val_F1_2), np.mean(val_F1_3), np.mean(val_F1_4), model, model_save_path)
        accs =np.mean(val_avg_acc)
        F1_1 = np.mean(val_F1_1)
        F1_2 = np.mean(val_F1_2)
        F1_3 = np.mean(val_F1_3)
        F1_4 = np.mean(val_F1_4)
        if early_stopping.early_stop:
            config.logger.info("Early stopping")
            accs = early_stopping.accs
            F1_1 = early_stopping.F1_1
            F1_2 = early_stopping.F1_2
            F1_3 = early_stopping.F1_3
            F1_4 = early_stopping.F1_4
            break
    result = f"train_losses: {train_losses};\n\
               val_losses: {val_losses};\n\
               train_accses: {train_accses};\n\
               val_accs: {val_accs};\n\
               accs:{accs},F1_1:{F1_1},F1_2:{F1_2},F1_3:{F1_3},F1_4:{F1_4};"
    config.logger.info(result)
    return train_losses,val_losses,train_accses,val_accs,accs,F1_1,F1_2,F1_3,F1_4

if __name__ == "__main__":
    config = Config()
    # model = BiGCN(in_feats = config.bert_embedding_dim,hid_feats = config.hidden_size,out_feats = config.out_size,dropout = config.dropout)
    # for iter in range(config.iterations):
    #     train_set, test_set = dataset_split(config.graph_dir+'/allP', config.cache_save_dir, config.validation_split)
    #     print(train_set, test_set)
    #     train_losses,val_losses,train_accs,val_accs = train_model(config, model, train_set, test_set, iter)