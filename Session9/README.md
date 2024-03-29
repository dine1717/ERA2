**Session  9 
**

Albumentations

<img width="563" alt="image" src="https://github.com/dine1717/ERA2/assets/73247157/4ea784ea-6bcd-4ffd-b9ac-ae3ee0a72c8b">


Model 


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()

# FIRST MAJOR BLOCK
        # first convolutional block
        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Using stride 2 without dilation instead of maxpooling
        self.convpool1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2), # in 32, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)   # to check if this can compensate for the mx pooling feature loss       
        )


# SECOND MAJOR BLOCK
          # first convolutional block
        self.convblock2a = nn.Sequential(
            # nn.Conv2d(in_channels = 16,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),                    # in 16, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16,groups = 16, dilation  = 1,padding = 1,kernel_size= (3,3)),
            nn.Conv2d(in_channels = 16,out_channels = 32, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3  
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # will the learning will be better in depth wise or in normal 3x3 convolutions?

        self.convblock2b = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 16, out 16, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Using stride 2 without dilation to simulate max pooling
        self.convpool2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2),
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)          # in 16, out 8, RF ?
        )

# THIRD MAJOR BLOCK
         # first convolutional block
        self.convblock3a = nn.Sequential(
            # nn.Conv2d(in_channels = 32,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 8, out 8, RF ?
            nn.Conv2d(in_channels = 32,out_channels = 64, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3           # in 8, out 8, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convblock3b = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 2,padding = 2, kernel_size= (3,3)),                     # in 8, out 8, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        # using dilation to simulate maxpooling
        self.convpool3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2),          # in 8, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)
        )

# FOURTH MAJOR BLOCK
        # second convolutional block
        self.convblock4a = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 64,out_channels = 64, groups = 64, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 128, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convblock4b = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 128,out_channels = 128, groups = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4,, RF ?
            nn.Conv2d(in_channels = 128,out_channels = 256, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )


# GAP LAYER
        self.gap = nn.AvgPool2d(4)                                                                                          # in 3, out 1, RF ?
        self.convblockf = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1a(x)
        x = self.convblock1b(x)
        x = self.convpool1(x)
        x = self.convblock2a(x)
        x = self.convblock2b(x)
        x = self.convpool2(x)
        x = self.convblock3a(x)
        x1 = self.convblock3b(x)
        x = torch.add(x,x1)
        x = self.convpool3(x)
        x = self.convblock4a(x)
        x = self.convblock4b(x)
        x = self.gap(x)
        x = self.convblockf(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



<img width="254" alt="image" src="https://github.com/dine1717/ERA2/assets/73247157/1b273ea3-446b-4657-9879-e1188196f08d">

1. Total 153104 Parametrs
2. Four Convolution blocks
3. Dilation layer in third block
4. Torch. Add layers in the third conv block
5. 3x3 convolution stride 2 followed by 1x1 convolution– max pool like layer
6. four depth wise convolutional layers
7. Target Accuracy – 85.39 (141 epoch)
8. Highest Accuracy – 86.28 (249 epoch)

Analysis and Findings of the architecture
1.Reason for normal 3x3 convolution layer following Depth wise convolution layer. A conventional 3x3 convolutional layer has been used in the first layer of every block and in all the layers of the fourth block. It is hypothesized that since depth wise convolution has lesser number of parameters and as initial extraction of features is important in the final prediction, this preliminary feature extraction process cannot be compromised. Lesser parameters means that lesser quality of feature extraction at the initial layers. Adding a normal 3x3 convolution following a depth-wise convolution ensures that there is an increase in parameters and hence the feature learning is not compromised

2.Addition of features from layer after the dilated kernel layer. The third convolutional block consists of two layers:- layer without dilation and layer with dilation which extracts same number of feature which same number of output dimension. Due to the dilation of kernel, there is a change in the pattern of feature extraction from the previously trained layers, hence may result in variation of validation accuracy of the model. To prevent this the layers are added using torch.add(). It is hypothesized that this will result in feature augmentation and hence better model performance than without feature addition from layers.

3.Adding a 1x1 pooling layer after the “max pool like” layer. Since there is no max pooling layer used here, a kernel of stride 2x2 will result in feature extraction with some features being missed out due to the stride. To compensate for this loss, the feature learning is augmented by using a 1x1 convolution. As 1x1 convolution sums up the features across channels to result in a new dimensional feature, this property may be exploited to is used as there is no max pooling layer. Hence to prevent loss of features 1x1 is used to add all the features that have been convolved separately

4.Torch.add () on normal layers. It was found that adding the feature output from same channel – same dimension output of two consecutive layers in the same convolutional block did not result in a significant increase in the performance of the model. However, removing Torch.add () from the convolutional block consisting of dilation layers resulted in fall of the performance of the model. This can be hypothesized that the way a feature needs to be extracted is to remain the same (i.e. gradual increase in receptive field) in all the layers. Any sudden increase in the receptive field size results in distortion of the learned features. Hence resulting in drop in performance. Adding the normal output to a dilated output restores this feature learning and results in better model performance.


