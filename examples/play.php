<?php

declare(strict_types=1);

error_reporting(E_ALL);
ini_set('display_errors', '1');
set_time_limit(0);

require_once __DIR__ . '/../vendor/autoload.php';

use HybridAgent\Agents\HybridAgent;
use HybridAgent\Core\Config;
use HybridAgent\Core\TicTacToe;
use HybridAgent\Neural\ContextEncoder;
use HybridAgent\Neural\DatasetGenerator;
use HybridAgent\Neural\ExperienceReplay;
use HybridAgent\Neural\MoveSampler;
use HybridAgent\Neural\Trainer;
use Rubix\ML\Datasets\Labeled;

// ==========================================
// 1. Load or train the model
// ==========================================
$encoder = new ContextEncoder(Config::HISTORY_LENGTH);
$trainer = new Trainer(Config::MODEL_PATH);

if ($trainer->modelExists()) {
    echo "--> Loading existing model weights from file (" . Config::MODEL_PATH . ")...\n";
    $estimator = $trainer->load();
} else {
    echo "1. Generating training dataset (context window: " . $encoder->featureCount() . " features)...\n";
    $generator = new DatasetGenerator($encoder);
    [$samples, $labels] = $generator->create(8000);
    $dataset = new Labeled($samples, $labels);
    echo "Dataset created with " . count($dataset) . " unique samples.\n\n";

    // Save a random subset of the original dataset as a rehearsal pool for future partial-training rounds
    $rehearsalIndices = (array) array_rand($samples, min(1000, count($samples)));
    $rehearsalData = [];
    foreach ($rehearsalIndices as $i) {
        $rehearsalData[] = ['context' => $samples[$i], 'move' => $labels[$i]];
    }
    file_put_contents(Config::REHEARSAL_POOL_PATH, json_encode($rehearsalData));

    echo "2. Training Neural Network with Rubix ML...\n";
    $startTime = microtime(true);
    $estimator = $trainer->trainNew($dataset);
    $time = round(microtime(true) - $startTime, 2);
    echo "Training completed in {$time} seconds!\n";

    $trainer->save($estimator);
    echo "--> Model saved successfully to " . Config::MODEL_PATH . "!\n";
}

// ==========================================
// 2. Main game loop with hybrid thinking layer + context window
// ==========================================
echo str_repeat("=", 45) . "\n";
echo "=== GAME STARTED: Human (X) vs Hybrid AI (O) ===\n";

$game = new TicTacToe();
$history = [];              // This game's move history (for the context window)
$aiDecisionPoints = [];     // Every AI decision point, stored as (context, Minimax's correct move)

$sampler = new MoveSampler(Config::CREATIVITY_CHANCE, Config::CREATIVITY_TEMPERATURE);
$agent = new HybridAgent($estimator, $encoder, $sampler, -1, Config::CONFIDENCE_THRESHOLD);

while ($game->checkWinner() === null) {
    $game->display();

    echo "Your move (X) [0-8]: ";
    $rawInput = fgets(STDIN);
    if ($rawInput === false) {
        break;
    }
    $inputPos = trim($rawInput);

    if ($inputPos === '' || !is_numeric($inputPos) || !$game->makeMove((int) $inputPos, 1)) {
        echo "--> Invalid move or spot taken! Try again.\n";
        continue;
    }
    $history[] = ['position' => (int) $inputPos, 'player' => 1];

    if ($game->checkWinner() !== null) {
        break;
    }

    $decision = $agent->decide($game, $history);

    $aiDecisionPoints[] = ['context' => $decision['context'], 'move' => $decision['correctMove']];

    $game->makeMove($decision['move'], -1);
    $history[] = ['position' => $decision['move'], 'player' => -1];

    if ($decision['isFallback']) {
        echo "--> AI (O) [Thinking Layer]: Low confidence (" . round($decision['confidence'] * 100, 1) . "%). Minimax took over, played position {$decision['move']}\n";
    } elseif ($decision['move'] !== $decision['topMove']) {
        echo "--> AI (O) [Creative]: Tried position {$decision['move']} instead of the top choice (Confidence in top choice: " . round($decision['confidence'] * 100, 1) . "%)\n";
    } else {
        echo "--> AI (O) [Neural Net]: Played position {$decision['move']} (Confidence: " . round($decision['confidence'] * 100, 1) . "%)\n";
    }
}

$game->display();
$winner = $game->checkWinner();
if ($winner === 1) {
    echo "Congratulations! You won!\n";
} elseif ($winner === -1) {
    echo "AI won the game!\n";
} else {
    echo "It's a draw!\n";
}

// ==========================================
// 3. Learning from this game (Experience Replay + Partial Training)
// ==========================================
$replay = new ExperienceReplay(Config::REPLAY_BUFFER_PATH);
foreach ($aiDecisionPoints as $point) {
    $replay->record($point['context'], $point['move']);
}
$replay->save();

$gamesSinceReplay = file_exists(Config::GAMES_SINCE_REPLAY_PATH)
    ? (int) file_get_contents(Config::GAMES_SINCE_REPLAY_PATH)
    : 0;
$gamesSinceReplay++;

echo "\n--> This game was added to the experience buffer. Games recorded since last training: $gamesSinceReplay/" . Config::REPLAY_TRIGGER_GAMES . "\n";

if ($gamesSinceReplay >= Config::REPLAY_TRIGGER_GAMES && $replay->count() > 0) {
    echo "--> Starting partial training on new experiences (with rehearsal to prevent forgetting)...\n";

    $rehearsalPool = file_exists(Config::REHEARSAL_POOL_PATH)
        ? json_decode(file_get_contents(Config::REHEARSAL_POOL_PATH), true)
        : [];
    $rehearsalSamples = array_column($rehearsalPool, 'context');
    $rehearsalLabels = array_column($rehearsalPool, 'move');
    $trainingBatch = $replay->buildTrainingBatch($rehearsalSamples, $rehearsalLabels, Config::REHEARSAL_SIZE);

    $trainer->partialTrain($estimator, $trainingBatch);

    echo "--> Model successfully updated and saved.\n";

    $replay->clear();
    $replay->save();
    file_put_contents(Config::GAMES_SINCE_REPLAY_PATH, '0');
} else {
    file_put_contents(Config::GAMES_SINCE_REPLAY_PATH, (string) $gamesSinceReplay);
}
