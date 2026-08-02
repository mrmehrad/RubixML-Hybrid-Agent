<?php

declare(strict_types=1);

namespace HybridAgent\Neural;

use HybridAgent\Core\TicTacToe;
use HybridAgent\Minimax\Minimax;

/**
 * Training dataset generation (with context support and style diversity).
 */
class DatasetGenerator
{
    public function __construct(private readonly ContextEncoder $encoder)
    {
    }

    /**
     * @return array{0: array, 1: array} [samples, labels]
     */
    public function create(int $sampleSize = 8000): array
    {
        $samples = [];
        $labels = [];
        $seen = [];

        for ($s = 0; $s < $sampleSize; $s++) {
            $game = new TicTacToe();
            $history = [];
            $randomMoves = rand(0, 6);
            $player = 1;

            // Style diversity for model "creativity": sometimes a fully random
            // opponent (wider state-space coverage), sometimes a semi-optimal
            // opponent (simulating different playing styles)
            $style = mt_rand(0, 100);

            for ($m = 0; $m < $randomMoves; $m++) {
                $avail = $game->getAvailableMoves();
                if (empty($avail) || $game->checkWinner() !== null) {
                    break;
                }

                if ($style < 60) {
                    $move = $avail[array_rand($avail)];
                } else {
                    $move = (mt_rand(0, 100) < 70)
                        ? Minimax::findBestMove($game, $player)
                        : $avail[array_rand($avail)];
                }

                $game->makeMove($move, $player);
                $history[] = ['position' => $move, 'player' => $player];
                $player = -$player;
            }

            if ($game->checkWinner() === null && !empty($game->getAvailableMoves())) {
                $bestMove = Minimax::findBestMove($game, $player);
                if ($bestMove !== -1) {
                    $context = $this->encoder->encode($game->getBoard(), $history, $player);
                    $key = implode(',', $context);

                    if (!isset($seen[$key])) {
                        $seen[$key] = true;
                        $samples[] = $context;
                        $labels[] = (string) $bestMove;
                    }
                }
            }
        }

        return [$samples, $labels];
    }
}
