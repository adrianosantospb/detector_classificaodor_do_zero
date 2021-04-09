import glob2
import os
from sklearn.model_selection import train_test_split 
import pandas as pd
import cv2
import numpy as np

'''
Funcoes auxiliares de pre-processamento dos dados.

Todas essas funcoes deverao ir para um arquivo de utilidade.
Lembre-se que todas as funcoes etc. estao de forma didaticas e que você DEVE otmizar o seu código uma vez que você entenda o passo a passo.
'''

'''
Com base em um diretório, um tipo de arquivo e se a busca deve ser recursiva, obtem todos os arquivos de um diretório base, gerando uma lista.
'''
def obtemTodosOsArquivos(diretorio_base, tipo_de_arquivo, eRecursivo=False):
    regra = "/**/*.{}" if eRecursivo else "/*.{}"
    caminho = diretorio_base + regra.format(tipo_de_arquivo)
    arquivos = glob2.glob(caminho , recursive=eRecursivo)
    return arquivos

'''
Obtem todas as anotacoes de um arquivo.
'''
def obtemLinhas(arquivo):
    with open(arquivo, "r") as f:
        return [l.strip() for l in f]

'''
Remove da lista de dados válidos as imagens que possuem mais de uma anotacao/objeto.
'''
def selecionarImagensComUmaAnotacao(arquivos_jpg):
    arquivos_final = []
    arquivos_removidos = []
    for arquivo in arquivos_jpg:
        linhas = obtemLinhas(arquivo.replace(".jpg", ".txt"))
        if len(linhas) == 1:
            arquivos_final.append(arquivo)
        else:
            arquivos_removidos.append(arquivo)
    return arquivos_final, arquivos_removidos

'''
Com base na lista de arquivos válidos, iremos gerar um dataframe para simplificar o nosso processo de manipulacao de dados.
Os dados no formato Darknet seguem a estrutura: classe x y w h, sendo (x,y) as coordenadas centrais do bbox anotado e (w,h) a largura e a altura da anotacao. Os valores sao normalizados por (W,H) da imagem.
'''
def gerandoDataFrame(arquivos_jpg):
    dataset = []
    for arquivo in arquivos_jpg:
        lista = obtemLinhas(arquivo.replace(".jpg", ".txt"))[0].split()
        dataset.append([arquivo, lista[0], lista[1], lista[2], lista[3], lista[4]])
    # Gera um dataframe com os valores da lista e atribue valores para as colunas.
    return pd.DataFrame(dataset, columns=["arquivos", "classes", "x", "y", "w", "h"])

'''
Agora, o dataframe será dividido em dados para treinamento e validacao do modelo com base no tamanho definido.
'''
def obtemDataSetsDeTreinamentoEValidacao(dataframe, tamanho=0.30):
    X = dataframe.drop('classes', axis=1)
    y = dataframe.classes

    # Obtem dados para treinamento e dados intermediarios
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=tamanho)
    
    # Treinamento, Validacao e teste
    return X_train, X_val, y_train, y_val  

'''
Funcao principal de obtencao de dados.
'''
def obtemDataFrame(diretorio_base, tipo, recursivo):
    arquivos_jpg = obtemTodosOsArquivos(diretorio_base, tipo, recursivo)
    arquivos_final, arquivos_removidos = selecionarImagensComUmaAnotacao(arquivos_jpg)
    print("Arquivos que serão utilizados: {}. Arquivos removidos: {}.".format(len(arquivos_final), len(arquivos_removidos)))
    return gerandoDataFrame(arquivos_final)
