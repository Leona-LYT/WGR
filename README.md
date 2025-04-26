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
sklearn==1.5.2
pyro==1.9.2
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
├── simulation/
│     ├── WGR_UniY.py
│     ├── WGR_MultiY.py
│     ├── select_m_UniY.py
│     ├── select_m_MultiY.py
│     ├── examples/
│         ├── WGR_M1.ipynb
│         ├── ...
│         └── WGR_M4.ipynb
├── real_data/
│     ├── RealData_CT.py
│     ├── RealData_UJI.py
│     ├── RealData_MNIST.py
│     ├── select_m_CT.py
│     ├── MNIST_CNN.ipynb
├── compared_methods/
│     ├── bnn_CT.py
│     ├── cqr_CT.py
```

## Components Descriptions
- **data:** Datasets used in this paper.
  1. **simulation data**: All simulation data can be generated using the script `SimulationData.py`.  
  2. **real data**:
     - **tabular datasets**: This includes the **CT slices dataset** and **UJIndoorLoc dataset**, both of which are included in the repository. Detailed information about these datasets can be found at the UCI Machine Learning Repository:
       (a). CT slices: [https://archive.ics.uci.edu/dataset/206/relative+location+of+ct+slices+on+axial+axis]
       (b). UJIIndoorLoc: [https://archive.ics.uci.edu/dataset/310/ujiindoorloc].
     - **image datasets**: Can be downloaded using the script `image_data.py`.
- **model:**  Contains the neural network architectures used in this paper.
  1. `generator.py` and `discriminator.py` contain the generator and discriminator networks used in this paper.
  2. `regression_net.py` contains the networks used in the Deep Nonparametric Least Squares (DNLS) method.
  3. `BNN.py`: contains the networks used in the Bayesian Neural Network methods (Jospin et al., 2022).
- **utils:**  Contains utility functions for running experiments.
  1. `basic_utils.py`: Basic utility functions.
  2. `training_utils.py`: Training routines for each method implemented in the paper. For real data analysis, the `model_type` parameter may be specified as either the dataset name or the keyword `real_data`.
  3. `evaluation_utilis.py`: Evaluation procedures for assessing method performance in both simulation studies and real data analyses.  
  4. `validation_utilis.py`: Validation routines for model selection and performance tracking.  
  5. `plot_utils.py`: Visualization utilities for generating figures used in the paper.
  6. 'cqr_utils.py`: Computes prediction intervals and corresponding coverage probabilities based on the method proposed by Romano et al. (2019), within the DNLS framework.
- **Simulation:** Contains code and examples for the simulation studies.  
  1. `WGR_UniY.py`: Provide the code for the simulation experiments with one-dimensional response Y.
  2. `WGR_MultiY.py`:Provide the code for the simulation experiments with multi-dimensional response Y.
  3. `select_m_UniY.py` and `select_m_MultiY.py`: Provide the code the selection of m in simulation studies.
  4. **examples**: Four Jupyter notebooks demonstrate how to run the proposed WGR method on the four simulation models considered in the paper (each notebook shows results from a single replication).
  5. Sensitivity analysis can be performed by changing the settings defined in the `argparse` configuration within the script.

- **real_data:** The experiments for the real data analysis can be conducted by the codes provided in this fold.
  1. `RealData_CT.py`: Provide the code for CT slices dataset. This code can handle real data analysis with one dimensional response Y.
  2. `RealData_UJI.py`: Provide the code for UJINdoorLoc dataset. This code can handle real data analysis with multi-dimensional response Y.
  3. `RealData_MNIST.py`: Provide the code for the reconstruction task of image data. Here, we use MNIST data as an example.
  4. `select_m_CT.py`: Provide the code for the selection of m in the analysis of the CT slice dataset.
  5. `MNIST_CNN.ipynb`: Provides an example of the reconstruction task using a CNN, with 2,000 samples used for training.
- **compared_methods:**
  1. `bnn_CT.py`: Provide the code for the BNN method used in the analysis of the CT slice dataset.
  2. `cqr_CT.py`: Provide the code for the CQR method used in the analysis of the CT slice dataset, which is employed to construct the prediction interval.
  3. **Note:** The compared cWGAN method can be reproduced using the WGR code with the settings: lambda_w = 1 and lambda_l = 0. 

## How to use it
### Workflow and Preparations
1. Install the PyTorch framework by following the official installation guide at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).  
2. Clone the repository and install the required Python modules.  
3. Run the experiments using the code provided in the `simulation/` and `real_data/` folders.

     
## 📚 References
Jospin, L. V., Laga, H., Boussaid, F., Buntine, W., & Bennamoun, M. (2022). Hands-on Bayesian neural networks—A tutorial for deep learning users. IEEE Computational Intelligence Magazine, 17(2), 29-48.

Liu, S., Zhou, X., Jiao, Y., & Huang, J. (2021). Wasserstein generative learning of conditional distribution. arXiv preprint arXiv:2112.10039.

Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized quantile regression. Advances in neural information processing systems, 32.


