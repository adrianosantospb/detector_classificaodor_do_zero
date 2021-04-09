import cv2
import time
import numpy as np
import torch
import glob
from utils.transformacao import letterbox, obtemImagem, converte_mascara_para_bbox, imprime_bbox, visualizar, cria_mascara
from torch.autograd import Variable


# Preparacao da imagem
arquivo = "./dataset/orange_72.jpg"
classes = ["Macã", "Laranja", "Banana"]

imagem = obtemImagem(arquivo)
bbox = np.array([0.519231, 0.506010, 0.850962, 0.810096], dtype=np.float32)

mascara = cria_mascara(imagem, bbox)

img = letterbox(imagem, 416)
img_bbox = letterbox(mascara, 416)

new_bbox = converte_mascara_para_bbox(img_bbox)

cv2.rectangle(img,(int(new_bbox[0]), int(new_bbox[1])), (int(new_bbox[2]), int(new_bbox[3])), (0,255,0) ,2)


cv2.imshow("Imagem", img)
cv2.imshow("Mascara", img_bbox)

cv2.waitKey(0)
cv2.destroyAllWindows()
