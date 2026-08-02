<?php

declare(strict_types=1);

namespace HybridAgent\Tests;

use HybridAgent\Neural\ContextEncoder;
use PHPUnit\Framework\TestCase;

final class ContextEncoderTest extends TestCase
{
    public function testFeatureCountMatchesHistoryLength(): void
    {
        $encoder = new ContextEncoder(4);
        // 9 board cells + 4 history slots * 9 cells each
        $this->assertSame(45, $encoder->featureCount());
    }

    public function testEncodeNormalizesPerspective(): void
    {
        $encoder = new ContextEncoder(1);
        $board = array_fill(0, 9, 0);
        $board[0] = -1; // O played the center-left cell

        // From O's perspective (-1), its own mark should normalize to +1
        $context = $encoder->encode($board, [], -1);

        $this->assertSame(1.0, $context[0]);
    }

    public function testEncodePadsEmptyHistory(): void
    {
        $encoder = new ContextEncoder(2);
        $board = array_fill(0, 9, 0);

        $context = $encoder->encode($board, [], 1);

        // 9 board features + 2 history slots (all zero-padded) * 9 = 27
        $this->assertCount(27, $context);
        $this->assertSame(array_fill(0, 27, 0.0), $context);
    }
}
