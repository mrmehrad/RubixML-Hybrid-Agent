<?php

declare(strict_types=1);

namespace HybridAgent\Tests;

use HybridAgent\Core\TicTacToe;
use HybridAgent\Minimax\Minimax;
use PHPUnit\Framework\TestCase;

final class MinimaxTest extends TestCase
{
    protected function setUp(): void
    {
        Minimax::resetCache();
    }

    public function testTakesWinningMoveWhenAvailable(): void
    {
        // X to move: X X _  -> must play position 2 to win
        $board = [1, 1, 0, -1, -1, 0, 0, 0, 0];
        $game = new TicTacToe($board);

        $this->assertSame(2, Minimax::findBestMove($game, 1));
    }

    public function testBlocksOpponentWinningMove(): void
    {
        // O to move: X X _  -> must block at position 2
        $board = [1, 1, 0, 0, -1, 0, 0, 0, -1];
        $game = new TicTacToe($board);

        $this->assertSame(2, Minimax::findBestMove($game, -1));
    }
}
