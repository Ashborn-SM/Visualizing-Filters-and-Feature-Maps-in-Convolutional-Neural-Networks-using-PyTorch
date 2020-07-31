# Visualizing-Filters-and-Feature-Maps-in-Convolutional-Neural-Networks-using-PyTorch

## About

This repository is about the code to extract cnn layers and visualise the filters and featuremaps

## Filters

Filters are set of weights which are learned using the backpropagation algorithm. If you do alot of practical deep learning coding, you may know filters in the name of kernels. Filter size can be of 3×3 or maybe 5×5 or maybe even 7×7. 
Filters in a CNN layer learn to detect abstract concepts like boundary of a face, edges of a buildings etc. By stacking more and more CNN layers on top of each other, we can get more abstract and in-depth information from a CNN.
![Filter](https://cdn-images-1.medium.com/max/1200/1*obE_Fc8k2LEcSIr057eubQ.png)

## FeatureMaps

Feature Maps are the results we get after applying the filter through the pixel value of the image.This is what the model see's in a image and the process is called convolution operation. The reason for visualising the feature maps is to gain deeper understandings about CNN.


![Feature maps](https://cdn-images-1.medium.com/max/800/1*dMsu9z5eP-aXZXXHkXP4vg.png)

## Extracting the CNN layers

```
conv_layers = []
model_weights = []
model_children = list(models.resnet50().children())
counter = 0

for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
```
The above code is simple and self-explanatory but it is limited to pre-existing models like other resnet model resnet-18, 34, 101, 152. For a custom model ,things will be different ,lets say there is a Sequential layer inside another Sequential layer and if there is a CNN layer it will be unchecked.
This is where the extractor.py module comes in. 

### Extractor class
The Extractor class can find every CNN layer(except down-sample layers) including their weights in any resnet model and almost in any custom resnet and vgg model. Its not limited to CNN layers it can find Linear layers and if the name of the Down-sampling layer is mentioned, it 
can find that too. It can also give some useful information like the number of CNN, Linear and Sequential layers in a model.

### How to use
In the Extractor class the model parameter takes in a model and the DS_layer_name parameter is optional. The DS_layer_name parameter is to find the down-sampling layer normally in resnet layer the name will be 'downsample' so it is kept as default.

```
extractor = Extractor(model = resnet, DS_layer_name = 'downsample')
```
The below code is to activate the program 
```
extractor.activate()
```
You can get relevant details in a dictionary by calling ```extractor.info()```
```
{'Down-sample layers name': 'downsample', 'Total CNN Layers': 49, 'Total Sequential Layers': 4, 'Total Downsampling Layers': 4, 'Total Linear Layers': 1, 'Total number of Bottleneck and Basicblock': 16, 'Total Execution time': '0.00137 sec'}
```
#### Accessing the weights and the layers
```
extractor.CNN_layers -----> Gives all the CNN layers in a model
extractor.Linear_layers --> Gives all the Linear layers in a model
extractor.DS_layers ------> Gives all the Down-sample layers in a model if there are any
extractor.CNN_weights ----> Gives all the CNN layer's weights in a model
extractor.Linear_weights -> Gives all the Linear layer's weights in a model

```
