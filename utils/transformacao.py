
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt 
import random
import torch

'''
Neste exemplo, iremos trabalhar com algumas funcoes de agumentacao de dados.
Como se trata de um exemplo puramente didático, selecionei apenas algumas operacoes básicas que serao utilizadas no processo de treinamento do modelo.
'''

'''
Funcao para obtencao de uma imagem a partir de um caminho 
'''
def obtemImagem(caminho_imagem):
    img = cv2.imread(str(caminho_imagem)).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
    return img

'''
Funcao auxiliar para visualizacao de imagem 
'''
def visualizar(imagem):
    plt.imshow(imagem.astype('uint8'))
    plt.show()


def imprime_bbox(imagem, bbox):
    cv2.rectangle(imagem,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0) ,2)
    
'''
Funcao de Letterbox 
'''
def letterbox(imagem, tamanho=416):
    tamanho_antigo = imagem.shape[:2]

    ratio = float(tamanho)/max(tamanho_antigo)
    novo_tamanho = tuple([int(x*ratio) for x in tamanho_antigo])

    im = cv2.resize(imagem, (novo_tamanho[1], novo_tamanho[0]))

    delta_w = tamanho - novo_tamanho[1]
    delta_h = tamanho - novo_tamanho[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    nova_imagem = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    
    return nova_imagem

'''
Funcao de normalizacao
Os valores utilizadoa aqui estao de acordo com o ImageNet
'''
def normaliza(imagem):
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (imagem - imagenet_stats[0])/imagenet_stats[1]

'''
Funcao de Flipping
# 1 = hor
# 0 = ver
# -1 = hor + ver
'''
def flipping(image, op=1):
    return cv2.flip(image, op)


'''
**********************************************************************
Funcoes relacionadas ao bbox 
**********************************************************************
'''
'''
Funcao auxiliar para obtencao das coordenadas de uma bbox de acordo com a imagem
'''
def fun_aux_para_obter_coord(imagem, bbox):
    (H, W) = imagem.shape[:2]
    
    # Get x and y base
    w = float(bbox[2])
    h = float(bbox[3])
    x = float(bbox[0]) 
    y = float(bbox[1])

    # Get x and y base
    x = x - w / 2
    y = y - h / 2

    # Update values according to image dimensions
    x = int(x * W)
    w = int(w * W)
    y = int(y * H)
    h = int(h * H)

    return x, y, w, h

'''
Funcao para criacao de uma mascara com base na imagem e no bbox.
'''
def cria_mascara(imagem, bbox):
    # Inicia array
    (H, W) = imagem.shape[:2]
    imagem_bbox = np.zeros((H, W))
    
    # Obtem as coordenadas d
    x, y, w, h = fun_aux_para_obter_coord(imagem, bbox)

    # Preenche a area
    imagem_bbox[y:y+h, x:x+w] = 255.

    return imagem_bbox

'''
Funcao de conversao de uma imagem/mascara para as coordenadas em bbox
'''
def converte_mascara_para_bbox(Y):
    rows, cols = np.nonzero(Y)
    if len(rows)==0: 
        return np.zeros(4, dtype=np.float32)
    y1 = np.min(rows)
    x1 = np.min(cols)
    y2 = np.max(rows)
    x2 = np.max(cols)
    return np.array([x1, y1, x2, y2], dtype=np.float32)

'''
**********************************************************************
'''


def transformacao(arquivo, bbox, ehTreinamento, tamanho):
    # Obtem imagem
    img = obtemImagem(arquivo)
    # Obtem mascara equivalente aa imagem
    img_bbox = cria_mascara(img, bbox)
    
    if ehTreinamento:
        # Flipping
        if (random.choice([True, False])):
            img = flipping(img)
            img_bbox = flipping(img_bbox)
    
    img = letterbox(img, tamanho)
    img_bbox = letterbox(img_bbox, tamanho)
    img = normaliza(img)
    img = np.rollaxis(img, 2)
    
    return img, converte_mascara_para_bbox(img_bbox) # converte as coordenadas x,y,w,h para x1,y1,x2,y2
