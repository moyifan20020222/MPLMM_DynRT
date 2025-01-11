import os

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report
from torch import nn
from src import model as mm
from src.utils import *
import torch.optim as optim
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


class MaskedCrossEntropyLoss(nn.Module):
    """

    错误的损失函数 是引起多分类失效的原因吗
    """

    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        # 考虑避免一下数据集的偏差 weight项， 是数据集中各个分类占比的比值 的反比
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    # weight=torch.tensor([2.59, 7.04, 31.76, 25.26, 6.63, 9.23, 6.46]).to(device)
    def forward(self, label, logits, x_v_mask):
        # label: [batch_size, conv_len]
        # logits: [batch_size, conv_len, num_class]
        batch_size, conv_len = label.shape
        label = torch.flatten(label).to(device)
        x_v_mask = torch.flatten(x_v_mask).to(device)
        # print("logits", logits)
        # print("label", label)
        logits = logits.contiguous().view(batch_size * conv_len, -1).to(device)
        # print("需要判断的两个", label.size(), logits.size())
        loss = self.loss_fn(logits, label)
        print("检测ignore_index是否屏蔽了填充项", loss, loss.size())
        # loss = loss[x_v_mask].mean()
        # loss = (loss * x_v_mask).sum() / x_v_mask.sum()
        return loss


def my_cal_prf1(pred_list, true_list, mask):
    """
    自已选取计算p,r,f1值，比如说不计算中立情绪
    三者都是一行很长的列表 且都是一个epoch的所有数据的预测值 真实值 和掩码
    """
    conf_matrix = np.zeros([7, 7])  # 初始化混淆矩阵
    pred_list = pred_list.astype(np.int32)
    # print("形状", pred_list.shape, true_list.shape)
    for i, item in enumerate(mask):
        if item == 1:
            conf_matrix[true_list[i]][pred_list[i]] += 1

    p = np.diagonal(conf_matrix / np.reshape(np.sum(conf_matrix, axis=0) + 1e-8, [1, 7]))
    r = np.diagonal(conf_matrix / np.reshape(np.sum(conf_matrix, axis=1) + 1e-8, [7, 1]))
    f = 2 * p * r / (p + r + 1e-8)
    weight = np.sum(conf_matrix, axis=1) / np.sum(conf_matrix)
    w_avg_f = np.sum(f * weight)
    return p, r, f, w_avg_f, weight


def my_cal_prf1_cause(pred_list, true_list, mask):
    """
    自已选取计算p,r,f1值，比如说不计算中立情绪
    三者都是一行很长的列表 且都是一个epoch的所有数据的预测值 真实值 和掩码
    """
    conf_matrix = np.zeros([2, 2])  # 初始化混淆矩阵
    pred_list = pred_list.astype(np.int32)
    # print("形状", pred_list.shape, true_list.shape)
    for i, item in enumerate(mask):
        if item == 1:
            conf_matrix[true_list[i]][pred_list[i]] += 1

    p = np.diagonal(conf_matrix / np.reshape(np.sum(conf_matrix, axis=0) + 1e-8, [1, 2]))
    r = np.diagonal(conf_matrix / np.reshape(np.sum(conf_matrix, axis=1) + 1e-8, [2, 1]))
    f = 2 * p * r / (p + r + 1e-8)
    weight = np.sum(conf_matrix, axis=1) / np.sum(conf_matrix)
    w_avg_f = np.sum(f * weight)
    return p, r, f, w_avg_f, weight


def initiate(hyp_params, train_loader, valid_loader, test_loader, audio_embedding, log):
    if hyp_params.pretrained_model == "PromptModel":
        model = getattr(mm, "PromptModel")(hyp_params)
        model = transfer_model(model, hyp_params.pretrained_model)
    elif hyp_params.pretrained_model == "MULTModel":
        model = getattr(mm, "MULTModel")(hyp_params)
    elif hyp_params.pretrained_model == 'DynRTMulTModel':
        model = getattr(mm, "DynRTMulTModel")(hyp_params)
    elif hyp_params.pretrained_model == 'PromptDynRTModel':
        model = getattr(mm, "PromptDynRTModel")(hyp_params)

    if hyp_params.use_cuda:
        model = model.to(device)

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    # criterion = getattr(nn, hyp_params.criterion)()
    criterion = MaskedCrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=hyp_params.when, factor=0.1, verbose=True
    )
    settings = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
    }
    if hyp_params.dataset == "ECF":
        return train_model(settings, hyp_params, train_loader, valid_loader, test_loader, audio_embedding, log)
    elif hyp_params.dataset == "ECF_cause":
        return train_model_cause(settings, hyp_params, train_loader, valid_loader, test_loader, audio_embedding, log)
    elif hyp_params.dataset == "RECCON":
        return test_RECCON(settings, hyp_params, train_loader, valid_loader, test_loader, audio_embedding, log)

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader, audio_embedding, log):
    """
    本部分只用于情绪识别的功能， 识别原因因为需要借助另一个修改的格式，其格式参考RECCON做了修改， 给定每一个情绪句子，再去找原因句子，并且
    可以顺带测试RECCON的数据集。
    """
    model = settings["model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    scheduler = settings["scheduler"]

    def train(model, optimizer, criterion, epoch):
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        ec_prediction_list = []
        ec_true_list = []
        ec_prediction_mask = []
        ec_total_sample = 0
        ec_loss = 0.
        start_time = time.time()
        for i_batch, data in enumerate(train_loader):
            token_ids, attention_mask, conversation_len, mask, adj_index, emo_label, cause_label, ece_pair, y_pairs, doc_id, x_v, x_v_mask, utter, speaker, sentence_mask = data
            # 因为token_ids,attention_mask 在没有经过roberta编码之前，是一个列表存储的，所以它的放入device的操作有所不同 sentence_mask指示每个对话每个句子实际内容
            token_ids = [t.to(device) for t in token_ids]
            attention_mask = [t.to(device) for t in attention_mask]
            conversation_len = [t.to(device) for t in conversation_len]
            mask.to(device)
            # adj_index.to(device) # 情绪识别任务中还暂时用不到他们
            emo_label.to(device)
            cause_label.to(device)
            # ece_pair.to(device)
            # y_pairs.to(device)
            doc_id.to(device)
            x_v.to(device)
            x_v_mask.to(device)

            model.zero_grad()

            batch_size = x_v_mask.data.sum().item()
            net = nn.DataParallel(model) if batch_size > 4 else model
            prediction = net(token_ids, attention_mask, conversation_len, x_v, x_v_mask, audio_embedding, sentence_mask)

            raw_loss = criterion(emo_label, prediction, x_v_mask)  # 三个函数都是二维张量，记录一个句子的情绪类型的预测值和真实值，并用x_v_mask指示真实部分
            raw_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            print("当前轮次的损失", raw_loss)
            proc_loss += raw_loss.item() * x_v_mask.data.sum().item()
            proc_size += x_v_mask.data.sum().item()
            prediction = torch.softmax(prediction, dim=2).to(device)
            prediction = torch.argmax(prediction, dim=2).to(device)
            print("预测", prediction)
            print("真实", emo_label)
            ec_sum = x_v_mask.data.sum().item()  # 记录这个batch中总的句子个数
            ec_loss = ec_loss + raw_loss.item() * ec_sum
            ec_prediction_list.append(torch.flatten(prediction.data).cpu().numpy())
            ec_true_list.append(torch.flatten(emo_label.data).cpu().numpy())
            ec_prediction_mask.append(torch.flatten(x_v_mask.data).cpu().numpy())
            ec_total_sample += ec_sum

            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print(
                    "Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}".format(
                        epoch,
                        i_batch,
                        num_batches,
                        elapsed_time * 1000 / hyp_params.log_interval,
                        avg_loss,
                    )
                )
                proc_loss, proc_size = 0, 0
                start_time = time.time()
        #  emotion
        ec_loss = ec_loss / ec_total_sample
        ec_prediction_list = np.concatenate(ec_prediction_list)
        ec_true_list = np.concatenate(ec_true_list)
        ec_prediction_mask = np.concatenate(ec_prediction_mask)

        # print("查看ec_pred_list信息维度， 然后write_data需要用它", ec_prediction_list, ec_prediction_list.shape)
        # print("试着取数据", ec_prediction_list[0])

        loss = torch.tensor(ec_loss, dtype=torch.float32).requires_grad_(True)

        print("一个epoch的整体损失", loss)

        # emotion
        precision, recall, f1, w_avg_f, weight = my_cal_prf1(ec_prediction_list, ec_true_list, ec_prediction_mask)

        # emotion
        f1_score_ec = f1_score(ec_true_list, ec_prediction_list, average='macro', sample_weight=ec_prediction_mask
                               , zero_division=0)

        # print("p,r的格式", precision, recall)
        # emotion
        precision = np.mean(precision)
        recall = np.mean(recall)
        # print("另一种计算得到的f1", f1)
        log_line = f'[Train] Epoch {epoch}: 情绪识别任务 loss: {round(ec_loss, 6)}, p_score: {precision}, r_score: {recall}' \
                   f', F1_score: {round(f1_score_ec, 4)}, F1_score2: {f1}, F1_score3: {w_avg_f} '
        print(log_line)
        log.write(log_line + '\n')

    def evaluate(model, log, valid_type="test"):
        model.eval()
        loader = test_loader if valid_type == "test" else valid_loader
        ec_prediction_list = []
        ec_prediction_mask = []
        ec_true_list = []
        ec_total_sample = 0
        ec_loss = 0.
        with torch.no_grad():
            for i_batch, data in enumerate(loader):
                token_ids, attention_mask, conversation_len, mask, adj_index, emo_label, cause_label, ece_pair, y_pairs, doc_id, x_v, x_v_mask, utter, speaker, sentence_mask = data
                # 因为token_ids,attention_mask 在没有经过roberta编码之前，是一个列表存储的，所以它的放入device的操作有所不同
                token_ids = [t.to(device) for t in token_ids]
                attention_mask = [t.to(device) for t in attention_mask]
                conversation_len = [t.to(device) for t in conversation_len]
                # print("conv_len", conversation_len)
                mask.to(device)
                # adj_index.to(device) # 情绪识别任务中还暂时用不到他们
                emo_label.to(device)
                cause_label.to(device)
                # ece_pair.to(device)
                # y_pairs.to(device)
                x_v.to(device)
                x_v_mask.to(device)
                doc_id.to(device)

                batch_size = x_v_mask.data.sum().item()
                net = nn.DataParallel(model) if batch_size > 4 else model
                prediction = net(token_ids, attention_mask, conversation_len, x_v, x_v_mask, audio_embedding,
                                 sentence_mask)

                loss = criterion(emo_label, prediction, x_v_mask)

                prediction = torch.softmax(prediction, dim=2).to(device)
                prediction = torch.argmax(prediction, dim=2).to(device)
                print("预测", prediction)
                print("真实", emo_label)

                ec_sum = x_v_mask.data.sum().item()  # 记录这个batch中总的句子个数
                ec_total_sample += ec_sum
                ec_loss = ec_loss + loss.item() * ec_sum
                x_v_mask_1 = torch.flatten(x_v_mask.data).cpu().numpy()
                emo_label_1 = torch.flatten(emo_label.data).cpu().numpy()
                prediction_1 = torch.flatten(prediction.data).cpu().numpy()

                # print("emo_label1",emo_label_1, emo_label_1.shape)
                # print("pred_label1",prediction_1, prediction_1.shape)
                # print("x_v_mask",x_v_mask_1, x_v_mask_1.shape)
                emo_label_mask = emo_label_1[np.where(x_v_mask_1 == 1)]
                prediction_mask = prediction_1[np.where(x_v_mask_1 == 1)]
                x_v_mask_2 = x_v_mask_1[np.where(x_v_mask_1 == 1)]
                # ec_prediction_list.append(torch.flatten(prediction.data).cpu().numpy())
                # ec_true_list.append(torch.flatten(emo_label.data).cpu().numpy())
                ec_prediction_list.append(prediction_mask)
                ec_true_list.append(emo_label_mask)
                # ec_prediction_mask.append(torch.flatten(x_v_mask.data).cpu().numpy())
                ec_prediction_mask.append(x_v_mask_2)
                ec_sample = x_v_mask.data.sum().item()
                ec_total_sample += ec_sample

        ec_loss = ec_loss / ec_total_sample
        ec_prediction_list = np.concatenate(ec_prediction_list)
        ec_true_list = np.concatenate(ec_true_list)
        ec_prediction_mask = np.concatenate(ec_prediction_mask)
        f1_score_ec = f1_score(ec_true_list, ec_prediction_list, average='macro', sample_weight=ec_prediction_mask,
                               zero_division=0)
        precision, recall, f1, w_avg_f1, weight = my_cal_prf1(ec_prediction_list, ec_true_list, ec_prediction_mask)
        precision = np.mean(precision)
        recall = np.mean(recall)
        print("\n情绪句子检测结果", precision, recall, f1_score_ec)
        log_line = f'[{valid_type}]: p_score: {precision}, r_score:{recall}, F1_score: {round(f1_score_ec, 4)}, F1_score2: {f1}, F1_score3: {w_avg_f1}' \
                   f', Weight:{weight}'
        print("检测真实值和预测值的维度", ec_true_list, ec_prediction_list)

        reports = classification_report(ec_true_list, ec_prediction_list,
                                        target_names=['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness',
                                                      'surprise'],
                                        digits=4, zero_division=0)

        print(log_line)
        log.write(log_line + '\n')
        # 不带权重修改的
        f1_scores = [f1_score_ec]
        # 带上权重修改的
        f1_scores = [w_avg_f1]
        model.train()

        return precision, recall, f1_scores, reports, ec_prediction_list, ec_true_list, ec_prediction_mask, ec_loss

    best_f1_score_ec = 0.
    best_f1_score_ec_test = 0.
    best_p_score_ec = 0.
    best_p_score_ec_test = 0.
    best_r_score_ec = 0.
    best_r_score_ec_test = 0.
    best_report_ec = None
    best_report_ec_test = None
    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train(model, optimizer, criterion, epoch)

        dev_precision, dev_recall, dev_f1_scores, dev_reports, dev_prediction_list, dev_true_list, dev_prediction_mask, dev_loss = evaluate(
            model, log, "dev")
        test_precision, test_recall, test_f1_scores, test_reports, test_prediction_list, test_true_list, test_prediction_mask, test_loss = evaluate(
            model, log, "test")

        end = time.time()
        duration = end - start
        scheduler.step(dev_loss)

        print("-" * 50)
        print(
            "Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}".format(
                epoch, duration, dev_loss, test_loss
            )
        )
        print("-" * 50)

        ec_final_f1_score_dev = dev_f1_scores[0]
        ec_final_f1_score_test = test_f1_scores[0]
        ec_final_p_score_dev = dev_precision
        ec_final_p_score_test = test_precision
        ec_final_r_score_dev = dev_recall
        ec_final_r_score_test = test_recall

        if dev_loss < best_valid:
            print(f"Saved model at {hyp_params.name}")
            path = "trained_model/" + hyp_params + "_" + hyp_params.missing_model + ".pt"
            torch.save(model, path)  # 记录模型
            best_valid = dev_loss
            best_f1_score_ec = ec_final_f1_score_dev
            best_f1_score_ec_test = ec_final_f1_score_test
            best_p_score_ec = ec_final_p_score_dev
            best_p_score_ec_test = ec_final_p_score_test
            best_r_score_ec = ec_final_r_score_dev
            best_r_score_ec_test = ec_final_r_score_test
            best_report_ec = dev_reports
            best_report_ec_test = test_reports

        log_line = f'[Epoch{epoch}--DEV--Emotion]: best_precision:{round(best_p_score_ec, 4)}, best_recall: {round(best_r_score_ec, 4)}, best_F1_score: {round(best_f1_score_ec, 4)}'
        print(log_line)
        log.write('\n\n' + log_line + '\n\n')
        log.write('Emotion:' + best_report_ec + '\n')

        log_line = f'[Epoch{epoch}--TEST--Emotion]: best_precision:{round(best_p_score_ec_test, 4)}, best_recall: {round(best_r_score_ec_test, 4)},best_F1_score: {round(best_f1_score_ec_test, 4)}'
        print(log_line)
        log.write('\n\n' + log_line + '\n\n')
        log.write('Emotion:' + best_report_ec_test + '\n')
        log.close()
    path = "trained_model/" + hyp_params.name + "_" + hyp_params.missing_model + ".pt"
    model = torch.load(path)
    precision, recall, f1_scores, reports, ec_prediction_list, ec_true_list, ec_prediction_mask, ec_loss = evaluate(
        model, log, "dev")

    result = f'[检验最好模型的验证集结果 -- Emotion]:  best_precision:{precision}, best_recall: {recall}, best_F1_score: {f1_scores}, report:{reports}'
    log.write(result + '\n')


def train_model_cause(settings, hyp_params, train_loader, valid_loader, test_loader, audio_embedding, log):
    """
    本部分只用于情绪识别的功能， 识别原因因为需要借助另一个修改的格式，其格式参考RECCON做了修改， 给定每一个情绪句子，再去找原因句子，并且
    可以顺带测试RECCON的数据集。
    """
    model = settings["model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    scheduler = settings["scheduler"]

    def train(model, optimizer, criterion, epoch):
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        ec_prediction_list = []
        ec_true_list = []
        ec_prediction_mask = []
        ec_total_sample = 0
        ec_loss = 0.
        start_time = time.time()
        for i_batch, data in enumerate(train_loader):
            token_ids, attention_mask, conversation_len, adj_index, cause_label, y_pairs, doc_id, x_v, x_v_mask, utter, speaker, sentence_mask = data
            # 因为token_ids,attention_mask 在没有经过roberta编码之前，是一个列表存储的，所以它的放入device的操作有所不同 sentence_mask指示每个对话每个句子实际内容
            token_ids = [t.to(device) for t in token_ids]
            attention_mask = [t.to(device) for t in attention_mask]
            conversation_len = [t.to(device) for t in conversation_len]
            cause_label.to(device)
            # ece_pair.to(device)
            # y_pairs.to(device)
            doc_id.to(device)
            x_v.to(device)
            x_v_mask.to(device)

            model.zero_grad()

            batch_size = x_v_mask.data.sum().item()
            net = nn.DataParallel(model) if batch_size > 4 else model
            prediction = net(token_ids, attention_mask, conversation_len, x_v, x_v_mask, audio_embedding, sentence_mask)

            raw_loss = criterion(cause_label, prediction, x_v_mask)  # 三个函数都是二维张量，记录一个句子的情绪类型的预测值和真实值，并用x_v_mask指示真实部分
            raw_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            print("当前轮次的损失", raw_loss)
            proc_loss += raw_loss.item() * x_v_mask.data.sum().item()
            proc_size += x_v_mask.data.sum().item()
            prediction = torch.softmax(prediction, dim=2).to(device)
            prediction = torch.argmax(prediction, dim=2).to(device)
            print("预测", prediction)
            print("真实", cause_label)
            ec_sum = x_v_mask.data.sum().item()  # 记录这个batch中总的句子个数
            ec_loss = ec_loss + raw_loss.item() * ec_sum
            ec_prediction_list.append(torch.flatten(prediction.data).cpu().numpy())
            ec_true_list.append(torch.flatten(cause_label.data).cpu().numpy())
            ec_prediction_mask.append(torch.flatten(x_v_mask.data).cpu().numpy())
            ec_total_sample += ec_sum

            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print(
                    "Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}".format(
                        epoch,
                        i_batch,
                        num_batches,
                        elapsed_time * 1000 / hyp_params.log_interval,
                        avg_loss,
                    )
                )
                proc_loss, proc_size = 0, 0
                start_time = time.time()
        #  emotion
        ec_loss = ec_loss / ec_total_sample
        ec_prediction_list = np.concatenate(ec_prediction_list)
        ec_true_list = np.concatenate(ec_true_list)
        ec_prediction_mask = np.concatenate(ec_prediction_mask)

        print("查看ec_pred_list信息维度， 然后write_data需要用它", ec_prediction_list, ec_prediction_list.shape)
        print("试着取数据", ec_prediction_list[0])

        loss = torch.tensor(ec_loss, dtype=torch.float32).requires_grad_(True)

        print("一个epoch的整体损失", loss)

        # cause
        precision, recall, f1, w_avg_f, weight = my_cal_prf1_cause(ec_prediction_list, ec_true_list, ec_prediction_mask)

        # cause
        f1_score_ec = f1_score(ec_true_list, ec_prediction_list, average='macro', sample_weight=ec_prediction_mask
                               , zero_division=0)

        # print("p,r的格式", precision, recall)
        # emotion
        precision = np.mean(precision)
        recall = np.mean(recall)
        # print("另一种计算得到的f1", f1)
        log_line = f'[Train] Epoch {epoch + 1}: 原因识别任务 loss: {round(ec_loss, 6)}, p_score: {precision}, r_score: {recall}' \
                   f', F1_score: {round(f1_score_ec, 4)}, F1_score2: {f1}, F1_score3: {w_avg_f} '
        print(log_line)
        log.write(log_line + '\n')

    def evaluate(model, log, valid_type="test"):
        model.eval()
        loader = test_loader if valid_type == "test" else valid_loader
        ec_prediction_list = []
        ec_prediction_mask = []
        ec_true_list = []
        ec_total_sample = 0
        ec_loss = 0.
        with torch.no_grad():
            for i_batch, data in enumerate(loader):
                token_ids, attention_mask, conversation_len, adj_index, cause_label, y_pairs, doc_id, x_v, x_v_mask, utter, speaker, sentence_mask = data
                # 因为token_ids,attention_mask 在没有经过roberta编码之前，是一个列表存储的，所以它的放入device的操作有所不同
                token_ids = [t.to(device) for t in token_ids]
                attention_mask = [t.to(device) for t in attention_mask]
                conversation_len = [t.to(device) for t in conversation_len]
                # print("conv_len", conversation_len)
                cause_label.to(device)
                # ece_pair.to(device)
                # y_pairs.to(device)
                x_v.to(device)
                x_v_mask.to(device)
                doc_id.to(device)

                batch_size = x_v_mask.data.sum().item()
                net = nn.DataParallel(model) if batch_size > 8 else model
                prediction = net(token_ids, attention_mask, conversation_len, x_v, x_v_mask, audio_embedding,
                                 sentence_mask)

                loss = criterion(cause_label, prediction, x_v_mask)

                prediction = torch.softmax(prediction, dim=2).to(device)
                prediction = torch.argmax(prediction, dim=2).to(device)

                ec_sum = x_v_mask.data.sum().item()  # 记录这个batch中总的句子个数
                ec_total_sample += ec_sum
                ec_loss = ec_loss + loss.item() * ec_sum
                x_v_mask_1 = torch.flatten(x_v_mask.data).cpu().numpy()
                emo_label_1 = torch.flatten(cause_label.data).cpu().numpy()
                prediction_1 = torch.flatten(prediction.data).cpu().numpy()

                # print("emo_label1",emo_label_1, emo_label_1.shape)
                # print("pred_label1",prediction_1, prediction_1.shape)
                # print("x_v_mask",x_v_mask_1, x_v_mask_1.shape)
                emo_label_mask = emo_label_1[np.where(x_v_mask_1 == 1)]
                prediction_mask = prediction_1[np.where(x_v_mask_1 == 1)]
                x_v_mask_2 = x_v_mask_1[np.where(x_v_mask_1 == 1)]
                # ec_prediction_list.append(torch.flatten(prediction.data).cpu().numpy())
                # ec_true_list.append(torch.flatten(emo_label.data).cpu().numpy())
                ec_prediction_list.append(prediction_mask)
                ec_true_list.append(emo_label_mask)
                # ec_prediction_mask.append(torch.flatten(x_v_mask.data).cpu().numpy())
                ec_prediction_mask.append(x_v_mask_2)
                ec_sample = x_v_mask.data.sum().item()
                ec_total_sample += ec_sample

        ec_loss = ec_loss / ec_total_sample
        ec_prediction_list = np.concatenate(ec_prediction_list)
        ec_true_list = np.concatenate(ec_true_list)
        ec_prediction_mask = np.concatenate(ec_prediction_mask)
        f1_score_ec = f1_score(ec_true_list, ec_prediction_list, average='macro', sample_weight=ec_prediction_mask,
                               zero_division=0)
        precision, recall, f1, w_avg_f1, weight = my_cal_prf1_cause(ec_prediction_list, ec_true_list, ec_prediction_mask)
        precision = np.mean(precision)
        recall = np.mean(recall)
        print("\n情绪句子检测结果", precision, recall, f1_score_ec)
        log_line = f'[{valid_type}]: p_score: {precision}, r_score:{recall}, F1_score: {round(f1_score_ec, 4)}, F1_score2: {f1}, F1_score3: {w_avg_f1}' \
                   f', Weight:{weight}'
        print("检测真实值和预测值的维度", ec_true_list.shape, ec_prediction_list.shape)

        reports = classification_report(ec_true_list, ec_prediction_list,
                                        target_names=['is not Cause', 'is Cause'],
                                        digits=4, zero_division=0)

        print(log_line)
        log.write(log_line + '\n')
        # 不带权重修改的
        f1_scores = [f1_score_ec]
        # 带上权重修改的
        f1_scores = [w_avg_f1]
        model.train()

        return precision, recall, f1_scores, reports, ec_prediction_list, ec_true_list, ec_prediction_mask, ec_loss

    best_f1_score_ec = 0.
    best_f1_score_ec_test = 0.
    best_p_score_ec = 0.
    best_p_score_ec_test = 0.
    best_r_score_ec = 0.
    best_r_score_ec_test = 0.
    best_report_ec = None
    best_report_ec_test = None
    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train(model, optimizer, criterion, epoch)

        dev_precision, dev_recall, dev_f1_scores, dev_reports, dev_prediction_list, dev_true_list, dev_prediction_mask, dev_loss = evaluate(
            model, log, "dev")
        test_precision, test_recall, test_f1_scores, test_reports, test_prediction_list, test_true_list, test_prediction_mask, test_loss = evaluate(
            model, log, "test")

        end = time.time()
        duration = end - start
        scheduler.step(dev_loss)

        print("-" * 50)
        print(
            "Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}".format(
                epoch, duration, dev_loss, test_loss
            )
        )
        print("-" * 50)

        ec_final_f1_score_dev = dev_f1_scores[0]
        ec_final_f1_score_test = test_f1_scores[0]
        ec_final_p_score_dev = dev_precision
        ec_final_p_score_test = test_precision
        ec_final_r_score_dev = dev_recall
        ec_final_r_score_test = test_recall

        if dev_loss < best_valid:
            print(f"Saved model at {hyp_params.name}")
            path = "trained_model/" + hyp_params.name + "_" + hyp_params.missing_model + ".pt"
            torch.save(model, path)  # 记录模型
            best_valid = dev_loss
            best_f1_score_ec = ec_final_f1_score_dev
            best_f1_score_ec_test = ec_final_f1_score_test
            best_p_score_ec = ec_final_p_score_dev
            best_p_score_ec_test = ec_final_p_score_test
            best_r_score_ec = ec_final_r_score_dev
            best_r_score_ec_test = ec_final_r_score_test
            best_report_ec = dev_reports
            best_report_ec_test = test_reports

        log_line = f'[Epoch{epoch}--DEV--Cause]: best_precision:{round(best_p_score_ec, 4)}, best_recall: {round(best_r_score_ec, 4)}, best_F1_score: {round(best_f1_score_ec, 4)}'
        print(log_line)
        log.write('\n\n' + log_line + '\n\n')
        log.write('Emotion:' + best_report_ec + '\n')

        log_line = f'[Epoch{epoch}--TEST--Cause]: best_precision:{round(best_p_score_ec_test, 4)}, best_recall: {round(best_r_score_ec_test, 4)},best_F1_score: {round(best_f1_score_ec_test, 4)}'
        print(log_line)
        log.write('\n\n' + log_line + '\n\n')
        log.write('Cause:' + best_report_ec_test + '\n')
        log.close()

    # TODO: 这里除了各个模态分别测试， 其实也可以 让所有缺失情况一起学习得到一个完整的模型， 比如说原先论文的方法就是 为每一个数据的缺失模态随机赋值，
    # TODO: 不过也可以用同一种缺失模态一起训练，那就是SMIL这样操作了
    path = "trained_model/" + hyp_params.name + "_" + hyp_params.missing_model + ".pt"
    model = torch.load(path)
    precision, recall, f1_scores, reports, ec_prediction_list, ec_true_list, ec_prediction_mask, ec_loss = evaluate(
        model, log, "dev")
    result = f'[检验最好模型的验证集结果 -- Cause]: best_precision:{precision}, best_recall: {recall}, best_F1_score: {f1_scores}, report:{reports}'
    log.write(result + '\n')


def test_RECCON(settings, hyp_params, train_loader, valid_loader, test_loader, audio_embedding, log):
    """
    本部分只用于情绪识别的功能， 识别原因因为需要借助另一个修改的格式，其格式参考RECCON做了修改， 给定每一个情绪句子，再去找原因句子，并且
    可以顺带测试RECCON的数据集。
    对于RECCON  只需要测试他在相对应的 训练好的缺失模态下的效果即可、
    """
    model = settings["model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    scheduler = settings["scheduler"]


    def evaluate(model, log, loader):
        model.eval()
        loader = loader
        ec_prediction_list = []
        ec_prediction_mask = []
        ec_true_list = []
        ec_total_sample = 0
        ec_loss = 0.
        with torch.no_grad():
            for i_batch, data in enumerate(loader):
                token_ids, attention_mask, conversation_len, adj_index, cause_label, y_pairs, doc_id, x_v, x_v_mask, utter, speaker, sentence_mask = data
                # 因为token_ids,attention_mask 在没有经过roberta编码之前，是一个列表存储的，所以它的放入device的操作有所不同
                token_ids = [t.to(device) for t in token_ids]
                attention_mask = [t.to(device) for t in attention_mask]
                conversation_len = [t.to(device) for t in conversation_len]
                # print("conv_len", conversation_len)
                cause_label.to(device)
                # ece_pair.to(device)
                # y_pairs.to(device)
                x_v.to(device)
                x_v_mask.to(device)
                doc_id.to(device)

                batch_size = x_v_mask.data.sum().item()
                net = nn.DataParallel(model) if batch_size > 8 else model
                prediction = net(token_ids, attention_mask, conversation_len, x_v, x_v_mask, audio_embedding,
                                 sentence_mask)

                loss = criterion(cause_label, prediction, x_v_mask)

                prediction = torch.softmax(prediction, dim=2).to(device)
                prediction = torch.argmax(prediction, dim=2).to(device)

                ec_sum = x_v_mask.data.sum().item()  # 记录这个batch中总的句子个数
                ec_total_sample += ec_sum
                ec_loss = ec_loss + loss.item() * ec_sum
                x_v_mask_1 = torch.flatten(x_v_mask.data).cpu().numpy()
                emo_label_1 = torch.flatten(cause_label.data).cpu().numpy()
                prediction_1 = torch.flatten(prediction.data).cpu().numpy()

                # print("emo_label1",emo_label_1, emo_label_1.shape)
                # print("pred_label1",prediction_1, prediction_1.shape)
                # print("x_v_mask",x_v_mask_1, x_v_mask_1.shape)
                emo_label_mask = emo_label_1[np.where(x_v_mask_1 == 1)]
                prediction_mask = prediction_1[np.where(x_v_mask_1 == 1)]
                x_v_mask_2 = x_v_mask_1[np.where(x_v_mask_1 == 1)]
                # ec_prediction_list.append(torch.flatten(prediction.data).cpu().numpy())
                # ec_true_list.append(torch.flatten(emo_label.data).cpu().numpy())
                ec_prediction_list.append(prediction_mask)
                ec_true_list.append(emo_label_mask)
                # ec_prediction_mask.append(torch.flatten(x_v_mask.data).cpu().numpy())
                ec_prediction_mask.append(x_v_mask_2)
                ec_sample = x_v_mask.data.sum().item()
                ec_total_sample += ec_sample

        ec_loss = ec_loss / ec_total_sample
        ec_prediction_list = np.concatenate(ec_prediction_list)
        ec_true_list = np.concatenate(ec_true_list)
        ec_prediction_mask = np.concatenate(ec_prediction_mask)
        f1_score_ec = f1_score(ec_true_list, ec_prediction_list, average='macro', sample_weight=ec_prediction_mask,
                               zero_division=0)
        precision, recall, f1, w_avg_f1, weight = my_cal_prf1_cause(ec_prediction_list, ec_true_list, ec_prediction_mask)
        precision = np.mean(precision)
        recall = np.mean(recall)
        print("\n情绪句子检测结果", precision, recall, f1_score_ec)
        log_line = f'[{loader}]: p_score: {precision}, r_score:{recall}, F1_score: {round(f1_score_ec, 4)}, F1_score2: {f1}, F1_score3: {w_avg_f1}' \
                   f', Weight:{weight}'
        print("检测真实值和预测值的维度", ec_true_list.shape, ec_prediction_list.shape)

        reports = classification_report(ec_true_list, ec_prediction_list,
                                        target_names=['is not Cause', 'is Cause'],
                                        digits=4, zero_division=0)

        print(log_line)
        log.write(log_line + '\n')
        # 不带权重修改的
        f1_scores = [f1_score_ec]
        # 带上权重修改的
        f1_scores = [w_avg_f1]
        model.train()

        return precision, recall, f1_scores, reports, ec_prediction_list, ec_true_list, ec_prediction_mask, ec_loss

    path = "trained_model/ECF_cause_" + hyp_params.missing_model + ".pt"
    model = torch.load(path)
    precision, recall, f1_scores, reports, ec_prediction_list, ec_true_list, ec_prediction_mask, ec_loss = evaluate(
        model, log, train_loader)
    result = f'[检验最好模型的训练集结果 -- Cause]: best_precision:{precision}, best_recall: {recall}, best_F1_score: {f1_scores}, report:{reports}'
    log.write(result + '\n')

    precision, recall, f1_scores, reports, ec_prediction_list, ec_true_list, ec_prediction_mask, ec_loss = evaluate(
        model, log, test_loader)
    result = f'[检验最好模型的测试集结果 -- Cause]: best_precision:{precision}, best_recall: {recall}, best_F1_score: {f1_scores}, report:{reports}'
    log.write(result + '\n')

    precision, recall, f1_scores, reports, ec_prediction_list, ec_true_list, ec_prediction_mask, ec_loss = evaluate(
        model, log, valid_loader)
    result = f'[检验最好模型的验证集结果 -- Cause]: best_precision:{precision}, best_recall: {recall}, best_F1_score: {f1_scores}, report:{reports}'
    log.write(result + '\n')


