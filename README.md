# football_rl

WARNING! This program can only be run on Windows.

Do not forget to create your own virtual environment before running this repo

Run below to install the requirements
```bash
pip install -r requirements.txt
```

Run the command below to run the training
```bash
python train.py
```

Run the command below to run test
```bash
python test.py
```
# Rules for the competition:

Players will progressively advance to the next tier if they win the match.

Each match consists of 2 rounds, the winner with the most goals will advance to the next tier. In the case of a draw, the player with the most reward points will advance to the next tier.

**Reward points Distribution:**

Kicking the ball forward = 10 points<br/>
Touching the ball backwards = -2.5 points<br/>
Spotting the ball in front =  0.5 points<br/>
Spotting the ball at the back = -0.25 points<br/>
Did not spot the ball = -0.2 points<br/>

Scoring a goal = 100 points<br/>
Opponent scoring a goal = -100 points<br/>

Existing penalty = -0.05 points per frame<br/>

GLHF
