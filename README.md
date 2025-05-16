
## Proximal Distilled Evolutionary Reinforcement Learning ##

This is modified version for PDERL algorithm.

#### original repo: https://github.com/crisbodnar/pderl
```
@inproceedings{bodnar2020proximal,
  title={Proximal distilled evolutionary reinforcement learning},
  author={Bodnar, Cristian and Day, Ben and Li{\'o}, Pietro},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={04},
  pages={3283--3290},
  year={2020}
}
```

#### UPDATE LOG ###
- update dependency gym to gymnasium


#### Run PDERL #### 

install dependencies:

```
pip install -r requirements.txt
```

To run PDERL with proximal mutations and distillation-based crossover use:

```bash
python run_pderl.py -env=$ENV_NAME$ -distil -proximal_mut -mut_mag=$MUT_MAG$ -logdir=$LOG_DIR$
```

#### ENVS TESTED #### 

'Hopper-v4' \
