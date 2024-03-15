
# Assignment-7
## Task
* Acheive 99.4% accuracy on MNIST dataset(this must be consistently shown in your last few epochs, and not a one-time achievement)
* Less than or equal to 15 Epochs
* Less than 8000 Parameters 
* Explain your 3 steps using these target, results, and analysis with links to your GitHub files

______________
# **Step 1**

[Code 1](https://github.com/dine1717/ERA2/blob/main/Session7/Step_1.ipynb)

# Target
 1. Settle on Architeture
 2. Visualize Data
 3. Define Data Loaders, Data Transformations and Image Normalization
 4. Design vanila model architeture.
 
# Result:
 1. Total Parameters: 147,344
 2. Best Training Accuarcy: 99.78
 3. Best Test Accuarcy: 99.07
 
# Analysis:
 1. Model is working but we see for some of the epochs its overfitting.
 2. Model will never reach 99.4 on test data as the train data already acheived a max of 99.78
 3. To avoid overfitting we need three things; 1.Drop0ut, 2. Fewer Parameters, 3. Batch normalization

___________

# **Step 2**

 [Code 2](https://github.com/dine1717/ERA2/blob/main/Session7/Step_2.ipynb)
 
 # Target
 1. Reduce Model Parameters
 2. Use Batch Normalization to improve accuracy
 3. Use Gap to improve accuracy
 
# Result:
 1. Total Parameters: 10616
 2. Best Training Accuarcy: 99.53
 3. Best Test Accuarcy: 99.48
 
# Analysis:
 1. Model is performing good and in some epochs it is close to the target.
 2. MNIST doesn't need 145k parameters even 10k is good enough
 3. Model is not overfitting, so we can reach 99.4 accuracy on test
 4. Data augmentation might help

___________

# **Step 3**
 
 [Code 3](https://github.com/dine1717/ERA2/blob/main/Session7/Step_3.ipynb)
 
# Target
1. Increase aacuracy  by using dropout and Augmentation
 
# Result:
 1. Total Parameters: 10616
 2. Best Training Accuarcy: 99.31
 3. Best Test Accuarcy: 99.42
 
# Analysis:
 1. The model attained 99.4 in 8th epoch but it is not consistent 
 2. Try going deeper by adding padding in the initial layers
 3. Add 1x1 conv block after GAP
___________

# **Step 4**

 [Code 4](https://github.com/dine1717/ERA2/blob/main/Session7/Step_4.ipynb)
 
# Target
1. Apply 1x1 after GAP
2. Add more layers 

# Result:
 1. Total Parameters: 8172
 2. Best Training Accuarcy: 98.73
 3. Best Test Accuarcy: 99.44
 
# Analysis:
 1. 8k parameters are good enough
 2. Going deeper with 7 conv layers helped
 3. Added 1x1 conv layer after GAP
 4. Consistently above 99.25 in the last 5 epochs

# Logs:

EPOCH: 8
Loss=0.006661698222160339 Batch_id=468 Accuracy=98.55: 100%|██████████| 469/469 [00:23<00:00, 19.60it/s]
Test set: Average loss: 0.0216, Accuracy: 9938/10000 (99.38%)

EPOCH: 9
Loss=0.052959173917770386 Batch_id=468 Accuracy=98.62: 100%|██████████| 469/469 [00:23<00:00, 19.71it/s]
Test set: Average loss: 0.0217, Accuracy: 9934/10000 (99.34%)

EPOCH: 10
Loss=0.04659676179289818 Batch_id=468 Accuracy=98.59: 100%|██████████| 469/469 [00:21<00:00, 21.35it/s]
Test set: Average loss: 0.0236, Accuracy: 9926/10000 (99.26%)

EPOCH: 11
Loss=0.008048984222114086 Batch_id=468 Accuracy=98.68: 100%|██████████| 469/469 [00:21<00:00, 21.41it/s]
Test set: Average loss: 0.0209, Accuracy: 9942/10000 (99.42%)

EPOCH: 12
Loss=0.06775439530611038 Batch_id=468 Accuracy=98.71: 100%|██████████| 469/469 [00:23<00:00, 20.27it/s]
Test set: Average loss: 0.0241, Accuracy: 9929/10000 (99.29%)

EPOCH: 13
Loss=0.046459171921014786 Batch_id=468 Accuracy=98.67: 100%|██████████| 469/469 [00:22<00:00, 20.75it/s]
Test set: Average loss: 0.0223, Accuracy: 9931/10000 (99.31%)

EPOCH: 14
Loss=0.02116444520652294 Batch_id=468 Accuracy=98.73: 100%|██████████| 469/469 [00:22<00:00, 20.61it/s]
Test set: Average loss: 0.0176, Accuracy: 9944/10000 (99.44%)
___________

# **Step 5**
 [Code 5](https://github.com/dine1717/ERA2/blob/main/Session7/Step_5.ipynb)
 
# Target
1. Less than 9000 parameters
2. Less than 15 epochs
3. Test with Cyclic LR
4. Add small dropout of 5%

# Results
1. Number of Parameters = 8172
2. Best Train Accuracy = 99.02
3.Best Test Accuracy = 99.57

# Analysis
1. Model is performing attained 99.48  10th epoch and max acciracy is 99.57
2.We pushed the model to achieve target with approx 8k Parameters
3. Model consistently has 99.4 accuracy in the last 5 epocs
4. Onecycle LR is pure magic

# Logs
EPOCH: 10
Loss=0.02343447133898735 Batch_id=937 Accuracy=98.73: 100%|██████████| 938/938 [01:20<00:00, 11.68it/s]
Epoch: 10 LR: [0.03171837652888333]

Test set: Average loss: 0.0184, Accuracy: 9948/10000 (99.48%)

EPOCH: 11
Loss=0.011960679665207863 Batch_id=937 Accuracy=98.77: 100%|██████████| 938/938 [01:19<00:00, 11.76it/s]
Epoch: 11 LR: [0.018813366974912007]

Test set: Average loss: 0.0177, Accuracy: 9944/10000 (99.44%)

EPOCH: 12
Loss=0.011525171808898449 Batch_id=937 Accuracy=98.92: 100%|██████████| 938/938 [01:21<00:00, 11.57it/s]
Epoch: 12 LR: [0.008679444396382234]

Test set: Average loss: 0.0160, Accuracy: 9948/10000 (99.48%)

EPOCH: 13
Loss=0.00202154740691185 Batch_id=937 Accuracy=98.97: 100%|██████████| 938/938 [01:20<00:00, 11.67it/s]
Epoch: 13 LR: [0.0022170522863831426]

Test set: Average loss: 0.0151, Accuracy: 9951/10000 (99.51%)

EPOCH: 14
Loss=0.05094844847917557 Batch_id=937 Accuracy=99.02: 100%|██████████| 938/938 [01:21<00:00, 11.47it/s]
Epoch: 14 LR: [4.0254362882487444e-07]

Test set: Average loss: 0.0150, Accuracy: 9957/10000 (99.57%)


# **Step 6**

##  *** Bonus ***

 
 [Code 6](https://github.com/dine1717/ERA2/blob/main/Session7/Step_6.ipynb)
 
# Target

1. Less than 7000 parameters
2. Less than 15 epochs
3. Test with Cyclic LR
4. Add small dropout of 5%

# Results

1. Number of Parameters = 6750
2. Best Train Accuracy = 98.98
3.Best Test Accuracy = 99.50

# Analysis

1.We pushed the model to achieve target with approx 6.7k Parameters
2. Model consistently has 99.4 accuracy in the last 6 epocs except one in between
3. 10 is the max output channels we have used except last conv block

# Logs
EPOCH: 12
Loss=0.0025025000795722008 Batch_id=468 Accuracy=98.85: 100%|██████████| 469/469 [00:21<00:00, 22.08it/s]
Epoch: 12 LR: [0.008670466465012771]
Test set: Average loss: 0.0186, Accuracy: 9941/10000 (99.41%)

EPOCH: 13
Loss=0.04146011918783188 Batch_id=468 Accuracy=98.95: 100%|██████████| 469/469 [00:21<00:00, 21.63it/s]
Epoch: 13 LR: [0.0022123586092353013]
Test set: Average loss: 0.0183, Accuracy: 9943/10000 (99.43%)

EPOCH: 14
Loss=0.03187761455774307 Batch_id=468 Accuracy=98.96: 100%|██████████| 469/469 [00:21<00:00, 22.00it/s]
Epoch: 14 LR: [4.101745150496986e-07]
Test set: Average loss: 0.0181, Accuracy: 9950/10000 (99.50%)
