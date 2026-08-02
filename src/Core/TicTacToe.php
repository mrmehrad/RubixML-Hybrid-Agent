<?php

declare(strict_types=1);

namespace HybridAgent\Core;

/**
 * Minimal Tic-Tac-Toe game engine.
 * Board is a flat array of 9 cells: 1 = X, -1 = O, 0 = empty.
 */
class TicTacToe
{
    private array $board;

    public function __construct(?array $board = null)
    {
        $this->board = $board ?? array_fill(0, 9, 0);
    }

    public function display(): void
    {
        $symbols = [1 => 'X', -1 => 'O', 0 => ' '];
        echo "\n";
        for ($i = 0; $i < 9; $i += 3) {
            echo " " . $symbols[$this->board[$i]] . " | " . $symbols[$this->board[$i + 1]] . " | " . $symbols[$this->board[$i + 2]] . "    (" . $i . "|" . ($i + 1) . "|" . ($i + 2) . ")\n";
            if ($i < 6) {
                echo "-----------  ---------\n";
            }
        }
        echo "\n";
    }

    public function makeMove(int $position, int $player): bool
    {
        if ($position < 0 || $position > 8 || $this->board[$position] !== 0) {
            return false;
        }
        $this->board[$position] = $player;
        return true;
    }

    public function getAvailableMoves(): array
    {
        $moves = [];
        foreach ($this->board as $index => $value) {
            if ($value === 0) {
                $moves[] = $index;
            }
        }
        return $moves;
    }

    public function checkWinner(): ?int
    {
        $lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6],
        ];

        foreach ($lines as $line) {
            $sum = $this->board[$line[0]] + $this->board[$line[1]] + $this->board[$line[2]];
            if ($sum === 3) {
                return 1; // X
            }
            if ($sum === -3) {
                return -1; // O
            }
        }

        if (!in_array(0, $this->board, true)) {
            return 0; // Draw
        }
        return null; // Ongoing
    }

    public function getBoard(): array
    {
        return $this->board;
    }
}
