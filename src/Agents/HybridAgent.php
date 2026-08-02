<?php

declare(strict_types=1);

namespace HybridAgent\Agents;

use HybridAgent\Core\Config;
use HybridAgent\Core\TicTacToe;
use HybridAgent\Minimax\Minimax;
use HybridAgent\Neural\ContextEncoder;
use HybridAgent\Neural\MoveSampler;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;

/**
 * HybridAgent ("thinking layer").
 *
 * Wraps a trained Rubix ML estimator and decides each move by:
 *  1. Asking the neural net for a probability distribution over moves.
 *  2. If confidence is high enough, allowing controlled creativity
 *     (Boltzmann exploration) among the neural net's own candidates.
 *  3. If confidence is too low, falling back to Minimax so the agent
 *     never plays an unsafe move purely out of uncertainty.
 *
 * Every decision point is returned along with the Minimax "correct" move,
 * so the caller can log it into Experience Replay regardless of what was
 * actually played.
 */
class HybridAgent
{
    public function __construct(
        private readonly PersistentModel $estimator,
        private readonly ContextEncoder $encoder,
        private readonly MoveSampler $sampler,
        private readonly int $player,
        private readonly float $confidenceThreshold = Config::CONFIDENCE_THRESHOLD,
    ) {
    }

    /**
     * @return array{
     *   move: int,
     *   context: array,
     *   correctMove: string,
     *   isFallback: bool,
     *   confidence: float,
     *   topMove: int
     * }
     */
    public function decide(TicTacToe $game, array $history): array
    {
        $context = $this->encoder->encode($game->getBoard(), $history, $this->player);

        $singleDataset = new Labeled([$context], ['?']);
        $probabilities = $this->estimator->proba($singleDataset)[0];

        $available = $game->getAvailableMoves();
        $bestScore = -1.0;
        $topMove = $available[0];

        foreach ($available as $moveIndex) {
            $prob = $probabilities[(string) $moveIndex] ?? 0.0;
            if ($prob > $bestScore) {
                $bestScore = $prob;
                $topMove = $moveIndex;
            }
        }

        $isFallback = false;

        if ($bestScore < $this->confidenceThreshold) {
            // If the neural net is unsure, Minimax takes over (no creativity
            // here — this path must always be safe)
            $move = Minimax::findBestMove($game, $this->player);
            $isFallback = true;
        } else {
            // When the model is confident, allow a bit of exploration among
            // reasonable options
            $move = $this->sampler->sample($probabilities, $available, $topMove);
        }

        // The "correct" answer for Experience Replay always comes from
        // Minimax, regardless of what the agent/creativity actually played
        $correctMove = $isFallback ? (string) $move : (string) Minimax::findBestMove($game, $this->player);

        return [
            'move' => $move,
            'context' => $context,
            'correctMove' => $correctMove,
            'isFallback' => $isFallback,
            'confidence' => $bestScore,
            'topMove' => $topMove,
        ];
    }
}
