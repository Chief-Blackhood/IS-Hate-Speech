## Instructions

### Clone the Repo
```bash
$ git clone https://github.com/Chief-Blackhood/IS-Hate-Speech.git
$ cd IS-Hate-Speech
```

### Setup Conda Environment
```bash
$ conda create -n pytorch_env
$ conda activate pytorch_env
$ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
$ conda install -c anaconda pip
$ pip install transformers
$ pip install click
$ pip install pyyaml
$ conda install numpy
$ conda install pandas
```
> Ensure pip is installed in conda environment by running `which pip`

> You might have to run `pip uninstall click` and `pip uninstall pyyaml` before installing again if it doesn't work

### Wandb Config

Create an account on [Wandb](https://wandb.ai/home), alternatively you can directly use my key which has been uploaded to [One Drive](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/shrey_gupta_students_iiit_ac_in/Ef4QzaOU8HhLubh6USyX3SEBIPZ-2BgkAw9U_s5Jf1ZW1g?e=6YjWcD)

```bash
$ conda activate pytorch_env
$ pip install wandb
$ wandb login
```

You should be all setup after this

### Train Model
```bash
$ conda activate pytorch_env
$ python3 model_comment/main.py --max_epochs 10
```

> You can run `python3 model_comment/main.py --help` to get a list of all the arguments which can be passed to the training script