from models.layers import LinearMerge, ConvMerge
import torch.nn as nn



def set_weight_align_param(models, args,train_weight_list):
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
                        named_layer_modules[module_i][1].train_weight_list.append(train_weight_list[module_j])
                        if args.bias:
                            named_layer_modules[module_i][1].bias_align_list.append(nn.Parameter(named_layer_modules[module_j][1].bias, requires_grad=True))
                    print(f'Model {module_i}, Layer {named_layer_modules[module_i][0]}: Added model {module_j}, layer {named_layer_modules[module_j][0]} weight ')

