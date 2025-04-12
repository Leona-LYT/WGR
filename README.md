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
├── Simulation/
│     ├── WGR_UnivariateY.py
│     ├── examples/
│         ├── WGR_M1.ipynb
│         ├── ...
│         └── WGR_M4.ipynb
``` 
- **data:** Datasets used in this paper.
  1. **simulation data**: All simulation data can be generated using the script `SimulationData.py`.  
  2. **real data**:
     - **tabular datasets**: This includes the **CT slices dataset** and **UJIndoorLoc dataset**, both of which are included in the repository. Detailed information about these datasets can be found at the UCI Machine Learning Repository.  
     - **image datasets**: Can be downloaded using the script `image_data.py`.
- **model:**  Contains the neural network architectures used in this paper.
- **utils:**  Contains utility functions for running experiments.
- **Simulation:** Contains code and examples of simulation studies:
  1. The simulation studies can be condcuted by using the scripts `WGR_Univarate.py` and `WGR_multivarate.py`.
  2. **examples**: Four jupyter notebooks show how to run the proposed WGR method for four simulation models considered in the paper (only one replication is shown here) 
  
## 📚 References
Jospin, L. V., Laga, H., Boussaid, F., Buntine, W., & Bennamoun, M. (2022). Hands-on Bayesian neural networks—A tutorial for deep learning users. IEEE Computational Intelligence Magazine, 17(2), 29-48.

Liu, S., Zhou, X., Jiao, Y., & Huang, J. (2021). Wasserstein generative learning of conditional distribution. arXiv preprint arXiv:2112.10039.

Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized quantile regression. Advances in neural information processing systems, 32.


