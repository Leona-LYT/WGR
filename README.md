# Wasserstein Generative Regression
The repository contains the code for the paper "Wasserstein Generative Regression"

## 📦 Python Module Versions
Our current implementation depends on the following Python packages:

```txt
numpy==2.1.3
pandas==2.2.3
scipy==1.14.1
torch==2.6.0+cu124
torchvision==0.21.0+cu124
argparse==1.1  
matplotlib==3.9.3
PIL==10.4.0
seaborn==0.13.2
```

## 📁 Project Structure 
The structure of this repository is as follows:
``` 
WGR/
├── README.md
├── data/
│     ├── CT.csv
│     ├── UJIndoorLocData.csv
│     ├── SimulationData.py
│     ├── image_data.py
|     ├── multivariateY.py
├── models/
│     ├── BNN.py
│     ├── discriminator.py
│     ├── generator.py
│     ├── regression_net.py
├── utils/
│     ├── basic_utils.py 
│     ├── cqr_utils.py
│     ├── evaluation_utils.py
│     ├── plot_utils.py
│     ├── training_utils.py
│     ├── validation_utils.py
``` 
- **data:** Original dataset used in training.
- **model:** Pretrained ResNet50 architecture.
- **output:** Prediction results are saved in `results/`.


