# EverybodyDanceNow
Motion Retargeting Video Subjects, Modified Version by stanleyshly
# Everybody Dance Now
This repo contains some bug fixes for the data preprocessing and inferencing scripts , as well a porting the code to work with Python 3, as before some code was not working properly. I also got the code to run on Google Colab, as most poeple don't have access to GPUs with enough Vram.
Currently, most of the code work on Google Colab, but the pose normalization code does not work, most of the other scripts are fully functioning.
I'm working on fixing it and/or replacing it with a completely new script.

## Training
Then run the Openpose file, but when creating the label and target images, run _only_ the graph_train.py cell, then flush and unmount with the last cell. 
Then, run Everybody Dance Now file, and train the global stage for approximately 5 epochs, then train the local stage for approximately 30 epochs, and the local stage with face generator and discriminator for around another 5 epochs. 
This will take a _long_ time due to a large required dataset, in my case, alternating google colab accounts, it took around 3 days. 

## Inferencing
To inference, repeat the same Openpose steps above, but use the graph_avesmooth.py cell instead of graph_train.py cell, and use the dance video instead of training video. 
Then run the first inferencing cell, as it include the face generator, and run the ffmpeg cell, but change the framerate to match the frame rate of the source dance video. 

## Acknowledgements

Original Repo adapted from [EverybodyDanceNow](https://github.com/carolineec/EverybodyDanceNow)

Model code adapted from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

Data Preparation code adapted from [Realtime_Multi-Person_Pose_Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

Data Preparation code based on outputs from [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
