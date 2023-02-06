import torch

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

tinymodel = TinyModel()
tinymodel2 = TinyModel()


for module in tinymodel.named_modules():
    print('here')
    n,m=module
    if not type(m) == torch.nn.Linear :
        continue
    print(module)
    break
    #n,m=module
    #if not type(m) == torch.nn.Linear :
        #continue
    #print(n)
    #print(tinymodel2[n])
print("hhdfadsfas")
x=[model.named_modules() for model in [tinymodel, tinymodel2]]
for module in zip(*x):

    #print(module[0])
    if not type(module[0][1]) == torch.nn.Linear :
        continue
    print('here2')
    print(module[0][0])
    #break