from models.layers import LinearMerge, ConvMerge
import torch.nn as nn

def set_weight_align_param_old(model1, model2, args):
    for model1_mods, model2_mods, in zip(model1.named_modules(), model2.named_modules(),):
        n1, m1 = model1_mods
        n2, m2 = model2_mods
        if not type(m2) == LinearMerge and not type(m2)==ConvMerge:
            continue
        if hasattr(m1, "weight"):
            '''
            m1.weight gets updated to m2.weight_align because it is not detached.  and vice versa
            This is a simple way to "share" the weights between models. 
            Alternatively we could set m1.weight=m2.weight_align after merge model is done training.  
            '''
            # We only want to merge one models weights in this file
            # m1.weight_align=nn.Parameter(m2.weight, requires_grad=True)s
            m2.weight_align = nn.Parameter(m1.weight, requires_grad=True)
            m1.weight_align = nn.Parameter(m2.weight, requires_grad=True)

            if args.bias:
                m2.bias_align = nn.Parameter(m1.bias, requires_grad=True)
                m1.bias_align = nn.Parameter(m2.bias, requires_grad=True)


def set_weight_align_param(models, args):


    print("HEEREErErEE")
    print(len(models))
    var = len(models)
    zipall = zip([model.named_modules() in models])

    for i in range(1, var) in zip(zipall):
        print(i)
        sys.exit()

    sys.exit()
    for model1_mods, model2_mods, in zip(model1.named_modules(), model2.named_modules(),):
        n1, m1 = model1_mods
        n2, m2 = model2_mods
        if not type(m2) == LinearMerge and not type(m2)==ConvMerge:
            continue
        if hasattr(m1, "weight"):
            '''
            m1.weight gets updated to m2.weight_align because it is not detached.  and vice versa
            This is a simple way to "share" the weights between models. 
            Alternatively we could set m1.weight=m2.weight_align after merge model is done training.  
            '''
            # We only want to merge one models weights in this file
            # m1.weight_align=nn.Parameter(m2.weight, requires_grad=True)s
            m2.weight_align = nn.Parameter(m1.weight, requires_grad=True)
            m1.weight_align = nn.Parameter(m2.weight, requires_grad=True)

            if args.bias:
                m2.bias_align = nn.Parameter(m1.bias, requires_grad=True)
                m1.bias_align = nn.Parameter(m2.bias, requires_grad=True)