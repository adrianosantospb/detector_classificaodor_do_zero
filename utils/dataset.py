from torch.utils.data import Dataset
from utils.transformacao import transformacao
from utils.util import obtemDataFrame, obtemDataSetsDeTreinamentoEValidacao
import numpy as np


class DatasetBase(Dataset):
    
    def __init__(self, tamanho, ehTreinamento=True, transformacao=True):
        
        diretorio_base = "./dataset"

        df = obtemDataFrame(diretorio_base, "jpg", True)
        X_train, X_val, y_train, y_val  = obtemDataSetsDeTreinamentoEValidacao(df)
        
        if ehTreinamento:
            descritores, labels = X_train, y_train
        else:
            descritores, labels = X_val, y_val  
        
        self.arquivos = descritores["arquivos"].values
        self.classes = np.array(labels.values, dtype=np.int)
        self.boxes = self._bounding_box(descritores)
        self.transformacao = transformacao
        self.tamanho = tamanho

    def __getitem__(self, index):    
        arquivo, bbox, classe = self.arquivos[index], self.boxes[index], self.classes[index]
        
        X, novo_bbox = transformacao(arquivo, bbox, self.transformacao, self.tamanho)

        return X, novo_bbox, classe
 
    def __len__(self):
        return len(self.arquivos)

    def _bounding_box(self,df):
        novo_bbox = []
        for _, row in df.iterrows():
            novo_bbox.append(np.array([row["x"], row["y"], row["w"], row["h"]], dtype=np.float32))
        return novo_bbox