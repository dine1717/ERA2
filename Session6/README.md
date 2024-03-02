# PART 1

# Train a Simple Neural Network using Microsoft Excel

### Neural Network Diagram
<img width="539" alt="image" src="https://user-images.githubusercontent.com/73247157/120009954-33663e80-bffa-11eb-98b3-d7e235a1724a.png">


> 1. Draw the neural network diagram as shown in the figure.
> 2. Connect all the neurons using arrows and mark them with appropriate names and values.

---

### FeedForward Equations
![image](https://user-images.githubusercontent.com/73247157/120010011-46790e80-bffa-11eb-8809-5283b09cb3c9.png)

> 1. Write all the feedforward equations using the above diagram as reference.
> 2. Use sigmoid as the activation function for the hidden and output layers.

---

### Backpropagation Equations
![image](https://user-images.githubusercontent.com/73247157/120010087-5c86cf00-bffa-11eb-913b-bdfaba575aee.png)


### Training the Neural Network
![image](https://user-images.githubusercontent.com/73247157/120010232-863ff600-bffa-11eb-9e09-27d118ed0aee.png)


> 1. Initialize all the values of the neurons and weights as shown in the neural network diagram.
> 2. Write the equations of the weights and their gradients using the above equations and choosing the right cell numbers.
> 3. Use a constant learning rate to update the weights. This will be changed to observe the effect of learning rate on loss during training.
---

### Loss Graphs
<img width="736" alt="image" src="https://user-images.githubusercontent.com/73247157/120010299-9a83f300-bffa-11eb-9549-ea526a8c2e27.png">


> ### Observation: 
> We can observe that higher the value of learning rate, higher the rate of convergence of loss for this particular problem. This is not true in most deep neural network problems as the learning rate is generally kept low to update the weights slowly. 


---
# Part-2
On MNIST data, create a model to get 
99.49% validation accuracy
Less than 20k Parameters
You can use anything from above you want. 
Less than 20 Epochs
Have used BN, Dropout, a Fully connected layer, have used GAP. 

### Final Results
1. Achieved 99.4 test accuracy in 16th epoch
2. Total Parameters 9848
3. Droput = 0.04
4. Augmentation = Yes
5. Optimizer = SGD 
6. BatchNorm  = Yes
7. 1x1 Con Layer = Yes, 2 times
8. Max Channels = 20


### Model Architecture
![image](https://user-images.githubusercontent.com/73247157/120020888-f6a14400-c007-11eb-8956-0e000a6386b2.png)


2 3x3 conv layers followed by 1 1x1 con layer
![image](https://user-images.githubusercontent.com/73247157/120021044-2fd9b400-c008-11eb-85a8-f5e657359850.png)


### Model Results
epoches: 1
  0%|          | 0/1875 [00:00<?, ?it/s]<ipython-input-8-0645169f52e4>:84: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
Loss=0.036472760140895844 Batch_id=1874 Accuracy=92.19: 100%|██████████| 1875/1875 [00:33<00:00, 56.18it/s]

Test set: Average loss: 0.0527, Accuracy: 9845/10000 (98.45%)

epoches: 2
Loss=0.016250338405370712 Batch_id=1874 Accuracy=97.24: 100%|██████████| 1875/1875 [00:33<00:00, 56.46it/s]

Test set: Average loss: 0.0357, Accuracy: 9887/10000 (98.87%)

epoches: 3
Loss=0.0073128510266542435 Batch_id=1874 Accuracy=97.77: 100%|██████████| 1875/1875 [00:33<00:00, 55.39it/s]

Test set: Average loss: 0.0275, Accuracy: 9910/10000 (99.10%)

epoches: 4
Loss=0.27955880761146545 Batch_id=1874 Accuracy=98.06: 100%|██████████| 1875/1875 [00:33<00:00, 55.72it/s]

Test set: Average loss: 0.0239, Accuracy: 9925/10000 (99.25%)

epoches: 5
Loss=0.007086154539138079 Batch_id=1874 Accuracy=98.22: 100%|██████████| 1875/1875 [00:33<00:00, 55.61it/s]

Test set: Average loss: 0.0251, Accuracy: 9923/10000 (99.23%)

epoches: 6
Loss=0.09488368034362793 Batch_id=1874 Accuracy=98.41: 100%|██████████| 1875/1875 [00:33<00:00, 55.60it/s]

Test set: Average loss: 0.0264, Accuracy: 9918/10000 (99.18%)

epoches: 7
Loss=0.05097135156393051 Batch_id=1874 Accuracy=98.42: 100%|██████████| 1875/1875 [00:33<00:00, 56.32it/s]

Test set: Average loss: 0.0187, Accuracy: 9940/10000 (99.40%)

epoches: 8
Loss=0.02279774285852909 Batch_id=1874 Accuracy=98.45: 100%|██████████| 1875/1875 [00:33<00:00, 55.80it/s]

Test set: Average loss: 0.0214, Accuracy: 9932/10000 (99.32%)

epoches: 9
Loss=0.04609740525484085 Batch_id=1874 Accuracy=98.55: 100%|██████████| 1875/1875 [00:33<00:00, 56.00it/s]

Test set: Average loss: 0.0230, Accuracy: 9919/10000 (99.19%)

epoches: 10
Loss=0.0074670505709946156 Batch_id=1874 Accuracy=98.50: 100%|██████████| 1875/1875 [00:33<00:00, 56.19it/s]

Test set: Average loss: 0.0230, Accuracy: 9923/10000 (99.23%)

epoches: 11
Loss=0.009039480239152908 Batch_id=1874 Accuracy=98.64: 100%|██████████| 1875/1875 [00:33<00:00, 56.43it/s]

Test set: Average loss: 0.0212, Accuracy: 9931/10000 (99.31%)

epoches: 12
Loss=0.01000845804810524 Batch_id=1874 Accuracy=98.68: 100%|██████████| 1875/1875 [00:33<00:00, 55.66it/s]

Test set: Average loss: 0.0227, Accuracy: 9930/10000 (99.30%)

epoches: 13
Loss=0.0005858689546585083 Batch_id=1874 Accuracy=98.73: 100%|██████████| 1875/1875 [00:33<00:00, 55.80it/s]

Test set: Average loss: 0.0209, Accuracy: 9930/10000 (99.30%)

epoches: 14
Loss=0.017353661358356476 Batch_id=1874 Accuracy=98.69: 100%|██████████| 1875/1875 [00:33<00:00, 55.66it/s]

Test set: Average loss: 0.0203, Accuracy: 9938/10000 (99.38%)

epoches: 15
Loss=0.007400782313197851 Batch_id=1874 Accuracy=98.76: 100%|██████████| 1875/1875 [00:33<00:00, 55.77it/s]

Test set: Average loss: 0.0197, Accuracy: 9932/10000 (99.32%)

epoches: 16
Loss=0.011832794174551964 Batch_id=1874 Accuracy=98.75: 100%|██████████| 1875/1875 [00:33<00:00, 55.49it/s]

Test set: Average loss: 0.0198, Accuracy: 9942/10000 (99.42%)

epoches: 17
Loss=0.010255944915115833 Batch_id=1874 Accuracy=98.83: 100%|██████████| 1875/1875 [00:33<00:00, 56.23it/s]

Test set: Average loss: 0.0201, Accuracy: 9935/10000 (99.35%)

epoches: 18
Loss=0.07478754222393036 Batch_id=1874 Accuracy=98.83: 100%|██████████| 1875/1875 [00:33<00:00, 56.20it/s]

Test set: Average loss: 0.0185, Accuracy: 9949/10000 (99.49%)

epoches: 19
Loss=0.0009058605646714568 Batch_id=1874 Accuracy=98.83: 100%|██████████| 1875/1875 [00:33<00:00, 56.50it/s]

Test set: Average loss: 0.0199, Accuracy: 9934/10000 (99.34%)
