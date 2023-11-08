
#### Cross-Silo Federated Learning across Divergent Domains with Iterative Parameter Alignment
Published at IEEE International Conference on Big Data

![alt text](teaser.png)


To get started, you can check out basic MNIST examples at: 
mnist_examples/mnist_mlp_2_.py and \
mnist_examples/mnist_mlp_2_detach.py\



You can run an experiment with the following code: 

```
python -u main.py --config=configs/cifar10_3_split_label_Conv4_5.yaml --gpu=2  --weight_seed=32 --seed=32   --same_initialization=True --random_topology=False --local_epochs=3 --merge_iter=3000
```


Example runs are in run_hist file with a nohup command. 