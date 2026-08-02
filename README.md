# RubixML Hybrid Agent

A hybrid Tic-Tac-Toe agent that combines a **RubixML neural network**
(`MultilayerPerceptron`) with a **Minimax** search fallback, a **context
window** encoder, and **Experience Replay** so the agent keeps improving
after every game against a human.

This repository is a refactor of a single-file prototype (`index.php`)
into a proper, testable, PSR-4 structure.

## Architecture

```
RubixML-Hybrid-Agent/
├── src/
│   ├── Agents/         # HybridAgent — the decision-making orchestrator
│   │                     (neural net + confidence gate + Minimax fallback + creativity)
│   ├── Neural/          # Context encoding, dataset generation, move sampling,
│   │                     experience replay, and the Trainer (model lifecycle)
│   ├── Minimax/         # Memoized Minimax search (ground-truth oracle + safety net)
│   └── Core/            # TicTacToe game engine + shared Config constants
├── tests/                # PHPUnit unit tests
├── data/                 # Generated: rehearsal pool, replay buffer, game counter
├── models/               # Generated: trained model weights (model.rbx)
├── examples/             # play.php — CLI entry point to play a full game
├── .gitignore
├── composer.json
└── README.md
```

### How a move is decided (`HybridAgent::decide`)

1. The current board + last N moves are encoded into a feature vector
   (`ContextEncoder`).
2. The RubixML `MultilayerPerceptron` returns a probability distribution
   over the 9 cells.
3. **If confidence ≥ threshold** (default `0.72`): a bit of controlled
   exploration is allowed via Boltzmann sampling (`MoveSampler`), so the
   agent doesn't always play the exact same move in identical positions.
4. **If confidence < threshold**: `Minimax` takes over completely — this
   path is always deterministic and safe.
5. Regardless of what was actually played, the *Minimax-correct* move for
   that position is recorded. After the game, all decision points are
   pushed into `ExperienceReplay`, and every `REPLAY_TRIGGER_GAMES` games
   the model is partially retrained on new experiences **plus** a
   rehearsal sample of old data, to avoid catastrophic forgetting.

## Requirements

- PHP >= 8.1
- [Composer](https://getcomposer.org/)

## Installation

```bash
composer install
```

## Usage

Play a game against the agent from the CLI:

```bash
composer play
# or
php examples/play.php
```

The first run will generate a synthetic training dataset (labeled by
Minimax) and train the model from scratch — this creates `models/model.rbx`
and `data/rehearsal_pool.json`. Subsequent runs load the saved model and
keep learning from your games via Experience Replay.

## Testing

```bash
composer test
# or
vendor/bin/phpunit tests
```

## Configuration

All tunable constants live in `src/Core/Config.php`:

| Constant                 | Meaning                                              |
|---------------------------|-------------------------------------------------------|
| `HISTORY_LENGTH`          | How many past moves are fed as context                |
| `CREATIVITY_CHANCE`       | Probability of deviating from the top move            |
| `CREATIVITY_TEMPERATURE`  | Boltzmann sampling temperature                         |
| `REPLAY_TRIGGER_GAMES`    | Partial-train the model after this many games          |
| `REHEARSAL_SIZE`          | How many old samples are replayed during partial training |
| `CONFIDENCE_THRESHOLD`    | Minimum neural-net confidence before trusting it over Minimax |
