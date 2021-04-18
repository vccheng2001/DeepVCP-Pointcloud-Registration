## DeepVCP: An End-to-End Deep Neural Network for Point Cloud Registration using Pytorch 

This is a Pytorch implementation of the DeepVCP (Virtual Corresponding Points) 
network for point cloud registration, based on the original paper
by Baidu Research https://songshiyu01.github.io/pdf/DeepVCP_W.Lu_S.Song_ICCV2019.pdf. 

The model aligns a source point cloud to a target point cloud by learning the rigid transform (translation + rotation matrix).

## Architecture 
The architecture consists of 
 1. Deep Feature Extraction Layer
 2. Point Weighting 
 3. Deep Feature Embedding
 4. CPG (Corresponding Point Generation) layer
