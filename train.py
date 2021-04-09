import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score
import random
from random import choices
from torch.autograd import Variable

from utils.dataset import DatasetBase
from modelo import AdrianoNet

# Modulos para auxilio na estrutura do projeto.
from tqdm import tqdm
import argparse
import logging
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np

import pycuda.driver as cuda
cuda.init()

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr

def main(parser):
    # ************************************ DADOS ***************************************************
    # Limpa o cache do CUDA
    torch.cuda.empty_cache()
    
    # Dataset de treinamento e validacao.
    trainamento = DatasetBase(parser.tamanho, True, True)
    validacao = DatasetBase(parser.tamanho, False, False)

    # Selecionar o dispositivo a ser utilizado (CPU ou GPU).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Obtem a proporcao de workers por GPU
    num_worker = 4 * int(torch.cuda.device_count())

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(dataset=trainamento, shuffle=True, batch_size=parser.batch_size,num_workers=num_worker, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validacao, batch_size=parser.batch_size)

    # ************************************* REDE ************************************************
    criterion = nn.CrossEntropyLoss()
    
    model = AdrianoNet(3)
    
    # GPU
    model.to(device)
    model.share_memory()
    model.half().float()

    # Otimizador
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    otimizador = torch.optim.Adam(parameters, lr=parser.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(otimizador, 200)

    best_acc = 0
    best_val_loss = 100

    # ************************************ TREINAMENTO E VALIDACAO ********************************************
    for epoch in range(1, parser.epochs):
        
        logging.info('Treinamento: {}'.format(str(epoch)))
        
        model.train()
        total = 0
        sum_loss = 0

        for step, (X, bboxReal, classeReal) in tqdm(enumerate(train_loader), total=len(train_loader)):
            
            otimizador.zero_grad()
            
            X = X.cuda().float()
            bboxReal = bboxReal.cuda().float()
            classeReal = classeReal.cuda()            
            
            # Obtem valores estimados
            preds = model(X)
            bboxPred, classePred = preds[:,:4], preds[:, 4:]


            loss_classe = F.cross_entropy(classePred, classeReal, reduction="sum")
            loss_bb = F.l1_loss(bboxPred, bboxReal, reduction="none").sum(1)
            loss_bb = loss_bb.sum()

            loss_total = loss_classe + (loss_bb/1000)
            loss_total.backward()
            
            otimizador.step()
            update_optimizer(otimizador, 0.001)

            total += step
            sum_loss += loss_total.item()

        train_loss = sum_loss/total

        print('Validacao: {}'.format(str(epoch)))
        
        model.eval()

        val_loss = 0
        acertos = 0
        
        with torch.no_grad():
            for step, (X, bboxReal, classeReal) in tqdm(enumerate(validation_loader), total=len(validation_loader)):
                
                otimizador.zero_grad()
                
                X = X.cuda().float()
                bboxReal = bboxReal.cuda().float()
                classeReal = classeReal.cuda()            
                
                # Obtem valores estimados
                preds = model(X)
                bboxPred, classePred = preds[:,:4], preds[:, 4:]

                loss_classe = F.cross_entropy(classePred, classeReal, reduction="sum")
            
                loss_bb = F.l1_loss(bboxPred, bboxReal, reduction="none").sum(1)
                loss_bb = loss_bb.sum()
            
                loss_val_total = loss_classe + (loss_bb/1000)

                bboxPred = torch.sigmoid(bboxPred)

                predito = torch.max(classePred,1)[1]
                acertos += (predito == classeReal).sum()

                val_loss += loss_val_total.detach().item()
            
            acc = np.round(float(acertos) / (len(validation_loader)*parser.batch_size), 4)
            
            # Imprime métricas
            if acc > best_acc or (acc >= best_acc and sum_loss <= best_val_loss):
                print("Um novo modelo foi salvo")
                # Nome do arquivo dos pesos
                pesos = "{}/{}.pt".format(parser.dir_save,str("best"))
                # Salvar os pesos
                chkpt = {'epoch': epoch,'model': model.state_dict()} 
                torch.save(chkpt, pesos)
                best_acc = float(acc)
            
            print("Train loss error: {:.4f}, Val loss error: {:.4f}, Best model:{:.4f} , Current model:{:.4f}".format(train_loss, val_loss, best_acc, acc))
        
        # Limpa o cache do CUDA
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dir_save', default="./pesos")
    parser.add_argument('--tamanho', type=int, default=416)
    
    parser = parser.parse_args()

    # Main function.
    main(parser)
