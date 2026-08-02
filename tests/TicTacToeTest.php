<?php

declare(strict_types=1);

namespace HybridAgent\Tests;

use HybridAgent\Core\TicTacToe;
use PHPUnit\Framework\TestCase;

final class TicTacToeTest extends TestCase
{
    public function testStartsEmpty(): void
    {
        $game = new TicTacToe();
        $this->assertCount(9, $game->getAvailableMoves());
        $this->assertNull($game->checkWinner());
    }

    public function testMakeMoveRejectsOccupiedCell(): void
    {
        $game = new TicTacToe();
        $this->assertTrue($game->makeMove(0, 1));
        $this->assertFalse($game->makeMove(0, -1));
    }

    public function testDetectsRowWin(): void
    {
        $game = new TicTacToe();
        $game->makeMove(0, 1);
        $game->makeMove(3, -1);
        $game->makeMove(1, 1);
        $game->makeMove(4, -1);
        $game->makeMove(2, 1);

        $this->assertSame(1, $game->checkWinner());
    }

    public function testDetectsDraw(): void
    {
        // X | O | X
        // X | O | O
        // O | X | X
        $board = [1, -1, 1, 1, -1, -1, -1, 1, 1];
        $game = new TicTacToe($board);

        $this->assertSame(0, $game->checkWinner());
    }
}
