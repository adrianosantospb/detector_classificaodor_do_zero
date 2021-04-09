import cv2
import time
import numpy as np
import torch
import glob
from utils.transformacao import letterbox, obtemImagem, normaliza, imprime_bbox, visualizar
from modelo import AdrianoNet
from torch.autograd import Variable


# Preparacao da imagem
arquivo = "./dataset/orange_72.jpg"
arquivo = "./dataset/orange_93.jpg" #duas laranjas
classes = ["Macã", "Laranja", "Banana"]

imagem = obtemImagem(arquivo)

img = letterbox(imagem, 416)

_img = normaliza(img)
_img = np.rollaxis(_img, 2)
_img = torch.from_numpy(_img).float() 
_img = _img.unsqueeze(0)

print(_img.shape)

# Instancia o modelo
weigths_files = glob.glob('./pesos/*.pt')
weights=weigths_files[0]

model = AdrianoNet(3)
model.eval()

if weights.endswith('.pt'):
    model.load_state_dict(torch.load(weights)['model'])

start_time = time.time()

# Predicao
with torch.no_grad():
    pred = model(_img)
    
    bboxPred, classePred = pred[:,:4], pred[:, 4:]
    clasPred = torch.max(classePred,1)[1]
    bboxPred = bboxPred.detach().numpy()
    
    cv2.putText(img, text=classes[clasPred.item()], org=(20,20),
            fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0),
            thickness=2, lineType=cv2.LINE_AA)
    
    imprime_bbox(img, bboxPred[0])

# Apresenta o resultado
cv2.imshow("Final", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("--- %s seconds ---" % (time.time() - start_time))
