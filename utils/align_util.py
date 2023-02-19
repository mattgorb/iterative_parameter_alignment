from models.layers import LinearMerge, ConvMerge
import torch.nn as nn



def set_weight_align_param(models, args):
    print('Aligning weights...')
    module_list=[model.named_modules() for model in models]
    for named_layer_modules in zip(*module_list):
        if not type(named_layer_modules[0][1]) == LinearMerge and not type(named_layer_modules[0][1])==ConvMerge:
            continue

        if hasattr(named_layer_modules[0][1], "weight"):
            for module_i in range(len(named_layer_modules)):
                for module_j in range(len(named_layer_modules)):
                    if module_j==module_i:
                        continue
                    else:

                        named_layer_modules[module_i][1].weight_align_list.append(nn.Parameter(named_layer_modules[module_j][1].weight, requires_grad=True))
                        if args.bias:
                            named_layer_modules[module_i][1].bias_align_list.append(nn.Parameter(named_layer_modules[module_j][1].bias, requires_grad=True))
                print(f'Layer {named_layer_modules[module_i][0]}: Added models to {module_i} ')




def set_weight_align_param_orig(models, args):
    for model1_mods, model2_mods, in zip(models[0].named_modules(), models[1].named_modules(),):
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

            print("Aligning...")
            m2.weight_align_list.append(nn.Parameter(m1.weight, requires_grad=True))
            m1.weight_align_list.append(nn.Parameter(m2.weight, requires_grad=True))

            if args.bias:
                m2.bias_align_list.append(nn.Parameter(m1.bias, requires_grad=True))
                m1.bias_align_list.append(nn.Parameter(m2.bias, requires_grad=True))
