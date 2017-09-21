import argparse
import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

import TVLoss

dtype = torch.cuda.FloatTensor

def image_loader(image_name, image_size):
    image = Image.open(image_name)
    w,h = image.size
    imsize = round(image_size * (w/h if h > w else h/w))
    loader = transforms.Compose([
        transforms.Scale(imsize), # scale so largest dim is image_size
        transforms.ToTensor()])
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image


def imshow(tensor, title=None):
    image = tensor.clone().cpu()
    image = image.squeeze()
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def imwrite(tensor, out_file):
    image = tensor.clone().cpu()
    image = image.squeeze().clamp(0,1)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    image.save(out_file)

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        #self.target = target.detach()
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        #self.loss =  * self.criterion(input, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss

class CovarianceMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d).clone()  # resise F_XL into \hat F_XL
        features = features - features.mean(1).unsqueeze(1) # center distribution

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        #        return G.div(features.size(1)-1) #G.div(a * b * c * d)
        # return G.div(a*b*c*d*(features.size(1)-1))
        #        return G.div(features.size(1)-1)
        return G.div(a * b * c * d)

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target, weight, gram_module=GramMatrix):
        super(StyleLoss, self).__init__()
        #self.target = target.detach()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = gram_module()
        self.gram = self.gram.cuda()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, style_img, content_img,
                               style_weight=1000, content_weight=1,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default,
                               gram_module=GramMatrix):
    cnn = copy.deepcopy(cnn)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    model_throwaway = nn.Sequential() # just to gather activations

    # move these modules to the GPU if possible:
    gram = gram_module()
    use_cuda = True
    if use_cuda:
        model = model.cuda()
        model_throwaway = model_throwaway.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)
            model_throwaway.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model_throwaway(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model_throwaway(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight, gram_module)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)
            model_throwaway.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model_throwaway(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model_throwaway(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight, gram_module)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***
            model_throwaway.add_module(name, layer)  # ***

            #    model.requires_grad=False
    return model, style_losses, content_losses

def get_input_param_optimizer(input_img,num_steps):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
#    optimizer = optim.Adam([input_param])
    optimizer = optim.LBFGS([input_param],max_iter=num_steps,tolerance_grad=-1,tolerance_change=-1)


    return input_param, optimizer

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps, print_iters,outpath,
                       style_weight, content_weight, tv_weight):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model1, style_losses1, content_losses1 = get_style_model_and_losses(cnn,
        style_img, content_img, style_weight, content_weight)
 #   model2, style_losses2, content_losses2 = get_style_model_and_losses(cnn,
 #       nn.AvgPool2d(2)(style_img), nn.AvgPool2d(2)(content_img), style_weight, content_weight)
 #   model3, style_losses3, content_losses3 = get_style_model_and_losses(cnn,
 #       nn.AvgPool2d(4)(style_img), nn.AvgPool2d(4)(content_img), style_weight, content_weight)
 #   model4, style_losses4, content_losses4 = get_style_model_and_losses(cnn,
 #       nn.AvgPool2d(8)(style_img), nn.AvgPool2d(8)(content_img), style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img,num_steps)

    tvloss = TVLoss.TVLoss(tv_weight)
    print('Optimizing..')
    run = [0]
    while run[0] < num_steps:
        print(run[0])
        def closure():
            # correct the values of updated input image
#            input_param.data.clamp_(0, 1)
            if run[0] % print_iters == 0:
                imwrite(input_param.data,outpath + str(run[0]) + '.png')
            #if run[0] % 100 == 0:
                #imshow(input_param.data)

            optimizer.zero_grad()
            model1(input_param)
  #          model2(nn.AvgPool2d(2)(input_param))
  #          model3(nn.AvgPool2d(4)(input_param))
  #          model4(nn.AvgPool2d(8)(input_param))
            style_score = 0
            content_score = 0
            tv_score = 0
            tv_loss = tvloss(input_param)

            for sl in style_losses1: #+ style_losses2 + style_losses3 + style_losses4:
                style_score += sl.backward()
            for cl in content_losses1: #+ content_losses2 + content_losses3 + content_losses4:
                content_score += cl.backward()
            tv_score = tvloss.backward()


            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.data[0], content_score.data[0]))
                print('TV Loss : {:4f}'.format(tv_score.data[0]))
                print()

            return style_score + content_score + tv_score

        optimizer.step(closure)

    # a last correction...
    input_param.data.clamp_(0, 1)

    return input_param.data

#cnn = models.vgg19(pretrained=True).features

def main():

    parser = argparse.ArgumentParser(description='Neural Style!', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--style_image', default='examples/inputs/seated-nude.jpg', help="Style targe image")
    parser.add_argument('--content_image', default='examples/inputs/tubingen.jpg', help="Content target image")
    parser.add_argument('--image_size', default=512, type=int, help="Maximum height / width of generated image")
    parser.add_argument('--content_weight', default=1, type=float, help=" ")
    parser.add_argument('--style_weight', default=1000, type=float, help=" ")
    parser.add_argument('--tv_weight', default=1e0, type=float, help=" ")
    parser.add_argument('--num_iterations', default=1000, type=int, help=" ")
    parser.add_argument('--output_image', default="out.png", help=" ")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Cuda is required, but not available")

    outpath, _ = os.path.splitext(args.output_image)

    style_image = image_loader(args.style_image, args.image_size).type(dtype)
    #plt.figure()
    #imshow(style_image.data)
    #print(style_image.size())

    content_image = image_loader(args.content_image, args.image_size).type(dtype)
    #plt.figure()
    #imshow(content_image.data)
    #print(content_image.size())

    input_image = Variable(torch.randn(content_image.data.size()) * 0.001).type(dtype)
    #plt.figure()
    #imshow(input_image.data)
    #print(input_image.size())

    plt.show()
    print("loading vgg19..")
    cnn = models.vgg19(pretrained=True).features
    print("loaded vgg19")
    use_cuda=True
    if use_cuda:
        print('using cuda')
        cnn = cnn.cuda()

    output = run_style_transfer(cnn,content_image,style_image,input_image,
                                num_steps=args.num_iterations,
                                print_iters=100,
                                outpath=outpath,
                                style_weight=args.style_weight,
                                content_weight=args.content_weight,
                                tv_weight=args.tv_weight)

    imwrite(output,args.output_image)
    return 0

if __name__ == '__main__':
    sys.exit(main())
