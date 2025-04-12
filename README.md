# Wasserstein Generative Regression
The repository contains the code for the paper "Wasserstein Generative Regression".

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
- **data:** Datasets used in this paper.
  1. **simulation data**: All simulation data can be generated using the script `SimulationData.py`.  
  2. **real data**:
     - **tabular datasets**: This includes the **CT slices dataset** and **UJIndoorLoc dataset**, both of which are included in the repository. Detailed information about these datasets can be found at the UCI Machine Learning Repository.  
     - **image datasets**: Can be downloaded using the script `image_data.py`.
- **model:**  Contains the neural network architectures used in this paper.
- **utils:**  Contains utility functions for running experiments.

## 📄 Reference
```bibtex
@article{liu2021wasserstein,
  title={Wasserstein generative learning of conditional distribution},
  author={Liu, Shiao and Zhou, Xingyu and Jiao, Yuling and Huang, Jian},
  journal={arXiv preprint arXiv:2112.10039},
  year={2021}
}

@article{romano2019conformalized,
  title={Conformalized quantile regression},
  author={Romano, Yaniv and Patterson, Evan and Candes, Emmanuel},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}

article{jospin2022hands,
  title={Hands-on Bayesian neural networks—A tutorial for deep learning users},
  author={Jospin, Laurent Valentin and Laga, Hamid and Boussaid, Farid and Buntine, Wray and Bennamoun, Mohammed},
  journal={IEEE Computational Intelligence Magazine},
  volume={17},
  number={2},
  pages={29--48},
  year={2022},
  publisher={IEEE}
}

