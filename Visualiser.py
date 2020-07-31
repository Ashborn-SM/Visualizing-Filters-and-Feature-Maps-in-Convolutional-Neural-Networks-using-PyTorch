# Importing libraries
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as t
import cv2 as cv
import torchvision.models as models
# Importing the module
from extractor import Extractor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Loading the model
resnet = models.resnet50()

model_children = list(resnet.children())

extractor = Extractor(model_children)
extractor.activate()

# Visualising the filters
plt.figure(figsize=(35, 35))
for index, filter in enumerate(extractor.CNN_weights[0]):
    plt.subplot(8, 8, index + 1)
    plt.imshow(filter[0, :, :].detach(), cmap='gray')
    plt.axis('off')

plt.show()


# Filter Map
img = cv.cvtColor(cv.imread('filtermap/5.png'), cv.COLOR_BGR2RGB)
img = t.Compose([
    t.ToPILImage(),
    t.Resize((128, 128)),
    # t.Grayscale(),
    t.ToTensor(),
    t.Normalize(0.5, 0.5)])(img).unsqueeze(0)

filtermaps = [extractor.CNN_layers[0](img)]
for x in range(1, len(extractor.CNN_layers)):
    filtermaps.append(extractor.CNN_layers[x](filtermaps[-1]))

# Visualising the filtermaps
for x in range(len(filtermaps)):
    plt.figure(figsize=(30, 30))
    layers = filtermaps[x][0, :, :, :].detach()
    for i, filter in enumerate(layers):
        if i == 64:
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis('off')
    # plt.savefig('filtermap%s.png'%(x))
    break
plt.show()
