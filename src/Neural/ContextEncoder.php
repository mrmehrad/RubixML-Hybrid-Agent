<?php

declare(strict_types=1);

namespace HybridAgent\Neural;

use HybridAgent\Core\Config;

/**
 * Context Encoder ("context window" layer).
 *
 * Instead of the model only seeing the current board, we append the last N
 * moves of the game so it can learn from the trajectory, not just the
 * snapshot.
 */
class ContextEncoder
{
    public function __construct(private readonly int $historyLength = Config::HISTORY_LENGTH)
    {
    }

    /**
     * @param array $board Current raw board state
     * @param array $history Chronological list of past moves: [['position' => int, 'player' => int], ...]
     * @param int $currentPlayer The player who is about to move (used to normalize perspective)
     */
    public function encode(array $board, array $history, int $currentPlayer): array
    {
        // Normalize: from the current player's point of view, "self" is always +1, "opponent" is always -1
        $normalizedBoard = array_map(fn ($v) => (float) ($v * $currentPlayer), $board);

        $historyFeatures = [];
        $recentHistory = array_slice($history, -$this->historyLength);
        $padCount = $this->historyLength - count($recentHistory);

        // Zero-pad if the history is shorter than historyLength
        for ($i = 0; $i < $padCount; $i++) {
            $historyFeatures = array_merge($historyFeatures, array_fill(0, 9, 0.0));
        }

        foreach ($recentHistory as $move) {
            $moveVector = array_fill(0, 9, 0.0);
            $moveVector[$move['position']] = (float) ($move['player'] * $currentPlayer);
            $historyFeatures = array_merge($historyFeatures, $moveVector);
        }

        return array_merge($normalizedBoard, $historyFeatures);
    }

    public function featureCount(): int
    {
        return 9 + 9 * $this->historyLength;
    }
}
