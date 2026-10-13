# FRONT and PI-FRONT: finite-fault rupture characterization from displacement waveforms

<img width="2913" height="1461" alt="figure_2" src="https://github.com/user-attachments/assets/90a2b7e9-a540-44bb-8ca7-c1bdaea63eaf" />

## Research scope
FRONT maps multistation, three-component, low-frequency displacement records to event moment magnitude and subfault strike-slip, dip-slip, rise-time, and rupture-onset fields. The network contains a spatiotemporal station graph encoder, a learned station-to-fault projection, and separate fault-graph branches for source parameters and rupture time. PI-FRONT adds a differentiable Green's-function waveform loss during training. The manuscript evaluates the Longmenshan fault system and transfer to the Jiangcuo fault.

## Scientific configuration recorded in the supplied code
The entry point of the code specifies 60 stations, 985 Longmenshan secondary faults, 512 samples per component, a 0.5-second sampling interval, and 20-meter, 35-second, and 260-second sliding/rising/rupture scales. Its training configuration specifies up to 350 training epochs, a batch size of 64, an initial AdamW learning rate of 0.001, and a precomputed Green's function tensor. The preprocessing driver requests a 70/20/10 random split and uses random state 130 for this split. These numbers describe the provided snapshot and should be checked against the experiment behind the submitted figure.

## How to Use
### 1. Create the Python environment
The GPU environment is **Ubuntu 22.04, Python 3.12, PyTorch 2.7.0, and CUDA 12.8**. A compatible NVIDIA GPU and driver are required. On Ubuntu 22.04, you can use [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) to create and activate an isolated environment from the repository root directory:
```bash
conda create -n front python=3.12 pip -y
conda activate front
```
Install the PyTorch 2.7.0 CUDA 12.8 build, then the remaining Python dependencies:
```bash
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```
### 2. Prepare and split the dataset
Place the dataset folder and the code folder in the same parent directory. Before preprocessing, update the dataset paths and processing options in run_processing.py to match your local files. Run the script from its containing directory:
```bash
python run_processing.py
```
The script creates a Split_info folder containing the files prepared for training. See the Split_info folder for the generated files.
### 3. Train the model
Set the training parameters in main.py for your experiment. After confirming that the required files are available in Split_info, run the following command from the directory containing main.py:
```bash
python main.py
```
## Contact information
For questions about the code or data, please contact the corresponding author:
Qifang Liu
Email:qifang_liu@126.com 
## License
The source code in this repository is licensed under the MIT License. 
## Acknowledgement
The network architecture draws on ideas from [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18), [Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet), [GAT](https://github.com/PetarV-/GAT), and [GCN](https://github.com/tkipf/gcn).
