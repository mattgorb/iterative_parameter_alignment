
#### Cross-Silo Federated Learning across Divergent Domains with Iterative Parameter Alignment
Published at IEEE International Conference on Big Data 2023

![alt text](teaser.png)


To get started, check out the basic MNIST examples: 
```
mnist_examples/mnist_mlp_2_.py 
mnist_examples/mnist_mlp_2_detach.py
```


The following code shows how to run an experiment with a specific configuration.  You can specify arguments in the config file or pass them in the python command.  

```
python -u main.py --config=configs/cifar10_3_split_label_Conv4_5.yaml --gpu=2  --weight_seed=32 --seed=32   --same_initialization=True --random_topology=False --local_epochs=3 --merge_iter=3000
```


Additional examples are in the run_hist file. 
