# Train Neural Networks

This repository provides a modular training environment for neural networks doing classification and regression tasks. It builds on [pytorch lightning](https://lightning.ai/docs/pytorch/stable/) to train and track the training process of neural networks. 

Different models and datasets can easily been added and accessed via [Hydra](https://hydra.cc/docs/intro/) which is an efficient library to handle different configuration settings.

## Requirements

Create a virtual environment with `python==3.10`. You can create a predefined `conda` environment with
```setup
conda env create -f utils/create_env.yml
```
Open the environment (e.g. `conda activate EiV`) and install the required packages via `pip`.  

```setup
pip install -r utils/requirements.txt
```

## Run the Code

Examples on how to run the code for specific settings are given in the folder `scripts`. Each bash script trains a specific neural network on the given dataset. 
1. Train the neural network (`train.py`)
2. The dictionary of the neural network can be transformed in canoncial form (standard keys) (`save_renamed_state_dict.py`)
3. A dataframe with predictions can be created (`test.py`)
4. Some visualizations can be plotted to obtain an intuition of the quality of the trained neural network (`plot_correlation.py`)


### Call scripts from subfolder

To run the script `enb` as a main script. Navigate to the main folder and run the command
```python 
python -m dataset.enb
```

## Disclaimer

The software is made available "as is" free of cost. The Author assumes no responsibility whatsoever for its use by other parties, and makes no guarantees, expressed or implied, about its quality, reliability, safety, suitability or any other characteristic. In no event will the Author be liable for any direct, indirect or consequential damage arising in connection


## Licence

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.