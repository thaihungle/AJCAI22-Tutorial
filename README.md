# AJCAI22-Tutorial
Demo code for AJCAI22-Tutorial

Install packages following requirements.txt  
```
pip install -r requirements.txt
```
### DQN
Follow dqn.ipynb  
Reference: https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb


### MBEC
- Create log and model folders to save training info 
```
mkdir log  
mkdir model 
```
- Run the script dqn_mbec.py using

```
python dqn_mbec.py --task MountainCar-v0 --rnoise 0.5  --render 0 --task2 mountaincar --n_epochs 100000 --max_episode 1000000 --model_name DTM --update_interval 100 --decay 1 --memory_size 3000 --k 15 --write_interval 10 --td_interval 1 --write_lr .5 --rec_rate .1 --rec_noise .1 --batch_size_plan 4 --rec_period 9999999999  --num_warm_up -1 --lr 0.0005   
```
- Monitor training using tensorboard
```
cd log
tensorboard --logdir=./
```
Reference: https://github.com/thaihungle/MBEC-plus  
