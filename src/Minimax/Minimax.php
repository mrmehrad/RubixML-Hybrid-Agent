<?php

declare(strict_types=1);

namespace HybridAgent\Minimax;

use HybridAgent\Core\TicTacToe;

/**
 * Minimax with memoization (deep-thinking / fallback layer).
 * Used both as the "ground truth" oracle for dataset generation and
 * Experience Replay labeling, and as a safety net when the neural net
 * is not confident enough during live play.
 */
class Minimax
{
    private static array $cache = [];

    public static function findBestMove(TicTacToe $game, int $player): int
    {
        $bestScore = ($player === 1) ? -INF : INF;
        $bestMove = -1;

        foreach ($game->getAvailableMoves() as $move) {
            $simGame = new TicTacToe($game->getBoard());
            $simGame->makeMove($move, $player);

            $score = self::minimax($simGame, 0, $player === -1);

            if ($player === 1) {
                if ($score > $bestScore) {
                    $bestScore = $score;
                    $bestMove = $move;
                }
            } else {
                if ($score < $bestScore) {
                    $bestScore = $score;
                    $bestMove = $move;
                }
            }
        }
        return $bestMove;
    }

    private static function minimax(TicTacToe $game, int $depth, bool $isMaximizing): int
    {
        $key = implode(',', $game->getBoard()) . ($isMaximizing ? '_1' : '_0');
        if (isset(self::$cache[$key])) {
            return self::$cache[$key];
        }

        $winner = $game->checkWinner();
        if ($winner !== null) {
            if ($winner === 1) {
                return 10 - $depth;
            }
            if ($winner === -1) {
                return $depth - 10;
            }
            return 0;
        }

        if ($isMaximizing) {
            $bestScore = -INF;
            foreach ($game->getAvailableMoves() as $move) {
                $simGame = new TicTacToe($game->getBoard());
                $simGame->makeMove($move, 1);
                $bestScore = max(self::minimax($simGame, $depth + 1, false), $bestScore);
            }
            self::$cache[$key] = $bestScore;
            return $bestScore;
        }

        $bestScore = INF;
        foreach ($game->getAvailableMoves() as $move) {
            $simGame = new TicTacToe($game->getBoard());
            $simGame->makeMove($move, -1);
            $bestScore = min(self::minimax($simGame, $depth + 1, true), $bestScore);
        }
        self::$cache[$key] = $bestScore;
        return $bestScore;
    }

    /**
     * Clears the static memoization cache. Mainly useful for tests that
     * need a clean slate between runs.
     */
    public static function resetCache(): void
    {
        self::$cache = [];
    }
}
