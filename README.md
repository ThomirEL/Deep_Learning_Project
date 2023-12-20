# Deep_Learning_Project
Deep Learning Project for Course 2456 DTU

To train model do python train.py <params>

For a list of params do python train.py --h

To test a model do python test.py <params>

To train a similar model like model 2 then run with params:
  Train dir = ./../data/train

  Valid dir = ./../data/valid

  Ckpt save path = ./../ckpts

  Ckpt overwrite = False

  Report interval = 16

  Train size = 128

  Valid size = 64

  Buffer size = 500

  Full = False

  Learning rate = 7e-05

  Adam = [0.9, 0.99, 1e-08]

  Batch size = 1

  Nb epochs = 50

  Loss = l1

  Cuda = True

  Plot stats = False

  Random flip = 0.8

