# Deep Learning Models for Stock Return Prediction
This repository contains codes for my quant research on utilizing different deep learning for stock return prediction based on alpha factors. How to apply shapley values to quantify factors' contribution in the deep learning models is also included. Alpha factors data are not included here for confidential reasons.

## File Descriptions:
+ `main.py`: The overall training process. Shapley calculation is also included here.
+ `model.py`: Different deep learning models written in Pytorch
+ `utils.py`: Auxiliary functions such as dataset processing and performance evaluation
+ `config.py`: configurations for models and training process
+ `BN_LSTM.py`: It is a rather independent file which displays how I build LSTM from scratch so that I can apply the Batch normalization method for LSTM discribed in this paper: [RECURRENT BATCH NORMALIZATION](https://arxiv.org/pdf/1603.09025.pdf)
