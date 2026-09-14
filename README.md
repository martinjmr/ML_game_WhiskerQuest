WhiskerQuest

A grid-world reinforcement learning environment where an agent (the mouse) must collect cheese while avoiding poison, under partial observability. Includes a Deep Q-Network (DQN) implementation trained from scratch with PyTorch, plus baseline agents for comparison.

The game
The board is a square grid (grid_size + 4, 17×17 by default) surrounded by a 2-cell impassable border.
At the start of each episode, the mouse spawns at a random position. Each other cell is randomly assigned cheese (+1 reward), poison (-1 reward), or nothing, with probability temperature (0.3 by default) — a cell never has both.
The mouse has 4 actions: up, down, left, right. Moving into the border makes it bounce back inward instead of leaving the grid.
Reward equals the value of the cell the mouse moves onto. Eating cheese or stepping on poison consumes that cell — it won't reappear.
The mouse only observes a local 5×5 window centered on itself, not the full board: this is a partially observable environment.
An episode ends after a fixed number of steps (max_time, 100 by default).
Optional exploration mode (--explore): a small penalty (0.1 per visit) is applied for revisiting already-visited cells, discouraging the agent from looping in place and encouraging it to explore new areas. This adds a third input channel (visit counts) to the state.
Agents
RandomAgent — picks a uniformly random action. Baseline for "no learning at all".
StraightAgent — always picks the same action. Baseline for "no strategy at all".
DQN — a Deep Q-Network trained with experience replay:
Replay buffer (Memory) storing (state, next_state, action, reward, done) transitions, sampled in batches for training.
Epsilon-greedy exploration, starting at 0.2 and decaying by a factor of 0.995 per epoch down to a floor of 0.01.
Standard Q-learning target with discount factor 0.99, target values clipped to [-3, 3] to stabilize training.
Two backbone options, selected via --agent:
fc: a fully-connected network (DenseModel) over the flattened 5×5 local observation.
cnn: a small convolutional network (CNNModel) over the same local observation treated as a multi-channel image.
Project structure
.
├── main.py            # Entry point: parses args, builds env/agent, runs training + eval
├── environment.py      # Game rules (grid, cheese/poison, movement, rewards)
├── agents.py            # Agent implementations (Random, Straight, DQN) + replay memory
├── models.py            # Neural network architectures (DenseModel, CNNModel)
├── train_utils.py       # Training and evaluation loops
├── utils.py              # Logging, checkpointing, GIF rendering, optimizer factory
└── experiments/          # Created at runtime: logs, saved models, plots, GIFs
Usage
bash
pip install torch numpy scipy imageio matplotlib pillow

python main.py \
  --path base \
  --agent cnn \
  --memory_size 1000 \
  --explore \
  --lr 0.1 \
  --epoch 300 \
  --optimizer SGD \
  --epoch_eval 10 \
  --freq 50

Key arguments:

Argument	Description	Default
--agent	random, straight, fc, or cnn	straight
--grid_size	Size of the playable grid (before border)	13
--temperature	Probability of cheese/poison per cell	0.3
--max_time	Max steps per episode	100
--explore	Enable the visit-penalty exploration bonus	False
--epoch	Number of training episodes	100
--lr	Learning rate	0.05
--optimizer	SGD, Adam, or RMSprop	SGD
--memory_size	Replay buffer size	1000
--batch_size	Replay batch size	32

During training, the model is periodically evaluated and checkpointed (every --freq epochs), and a GIF of a played episode is saved to experiments/<path>/GIF/.

Output

Each run creates an experiments/<path>/ folder containing:

info.log — training log (if --log_type file)
models/ — saved model checkpoints (.pt)
plots/ — individual frame images per episode
GIF/ — animated GIF of a full episode, cheese in red, poison in blue, the mouse in white
Possible improvements
Tune temperature and grid_size to study how cheese/poison density affects learning difficulty.
Compare fc vs cnn backbones on sample efficiency.
Add a target network / double DQN to reduce Q-value overestimation.
