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
│     ├── WGR_UnivariateY.py
│     ├── examples/
│         ├── WGR_M1.ipynb
│         ├── ...
│         └── WGR_M4.ipynb
├── real_data/
│     ├── RealData_UniY.py
│     ├── RealData_MultiY.py
│     ├── RealData_Image.py
```

## Components Descriptions
- **data:** Datasets used in this paper.
  1. **simulation data**: All simulation data can be generated using the script `SimulationData.py`.  
  2. **real data**:
     - **tabular datasets**: This includes the **CT slices dataset** and **UJIndoorLoc dataset**, both of which are included in the repository. Detailed information about these datasets can be found at the UCI Machine Learning Repository.  
     - **image datasets**: Can be downloaded using the script `image_data.py`.
- **model:**  Contains the neural network architectures used in this paper.
  1. `generator.py` and `discriminator.py` contain the generator and discriminator networks used in this paper.
  2. `regression_net.py` contains the networks used in the Deep Nonparametric Least Squares (DNLS) method.
  3. `BNN.py`: contains the networks used in the Bayesian Neural Network methods (Jospin et al., 2022).
- **utils:**  Contains utility functions for running experiments.
  1. `basic_utils.py': Basic utility functions.
  2. `training_utils.py': Training routines for each method implemented in the paper. For real data analysis, the `model_type` parameter may be specified as either the dataset name or the keyword `real_data`.
  3. `evaluation_utilis.py': Evaluation procedures for assessing method performance in both simulation studies and real data analyses.  
  4. `validation_utilis.py': Validation routines for model selection and performance tracking.  
  5. `plot_utils.py': Visualization utilities for generating figures used in the paper.
  6. 'cqr_utils.py': Computes prediction intervals and corresponding coverage probabilities based on the method proposed by Romano et al. (2019), within the DNLS framework.
- **Simulation:** Contains code and examples for the simulation studies.  
  1. The simulation experiments can be conducted using the scripts `WGR_Univariate.py` and `WGR_Multivariate.py`.  
  2. **examples**: Four Jupyter notebooks demonstrate how to run the proposed WGR method on the four simulation models considered in the paper (each notebook shows results from a single replication).
  3. The dimensionality of the noise vector \( \eta \) can be selected by using the function `selection_m` provided in the script `basic_utils.py` located in the `utils` folder. An example demonstrating the selection procedure is given in `select_m.py` in the `Simulation` folder.
  4. Sensitivity analysis can be performed by changing the settings defined in the `argparse` configuration within the script.
  5.  **Note:** The compared cWGAN method can be reproduced using the WGR code with the settings: lambda_w = 1 and lambda_l = 0.
- **Real data:** The experiments for the real data analysis can be conducted by the codes provided in this fold.
  1. `RealData_UniY.py`: Provide the code for real data with one dimensional response Y.
  2. `RealData_MultiY.py`: Provide the code for real data with multi-dimensional response Y.
  3. `RealData_Image.py`: Provide the code for the reconstruction task of image data.

### Workflow and Preparations
1. Install the PyTorch framework by following the official installation guide at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).  
2. Clone the repository and install the required Python modules.  
3. Run the experiments using the code provided in the `simulation/` and `real_data/` folders.
   
     
## 📚 References
Jospin, L. V., Laga, H., Boussaid, F., Buntine, W., & Bennamoun, M. (2022). Hands-on Bayesian neural networks—A tutorial for deep learning users. IEEE Computational Intelligence Magazine, 17(2), 29-48.

Liu, S., Zhou, X., Jiao, Y., & Huang, J. (2021). Wasserstein generative learning of conditional distribution. arXiv preprint arXiv:2112.10039.

Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized quantile regression. Advances in neural information processing systems, 32.


