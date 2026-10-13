<?php

declare(strict_types=1);

namespace HybridAgent\Core;

/**
 * Centralized configuration for the hybrid agent system.
 * Kept as class constants (instead of global consts) so it can be
 * autoloaded, referenced explicitly (Config::HISTORY_LENGTH), and
 * mocked/overridden in tests if ever needed.
 */
final class Config
{
    // How many past moves to feed the model as context
    public const HISTORY_LENGTH = 6;

    // Probability the model tries something other than its top move
    public const CREATIVITY_CHANCE = 0.05;

    // Boltzmann sampling temperature; >1 = more variety, <1 = more conservative
    public const CREATIVITY_TEMPERATURE = 0.8;

    // Partial-train the model after this many games
    public const REPLAY_TRIGGER_GAMES = 5;

    // How many old samples to replay during partial training (prevents catastrophic forgetting)
    public const REHEARSAL_SIZE = 500;

    // Minimum neural-net confidence required before trusting it over Minimax
    public const CONFIDENCE_THRESHOLD = 0.85;

    // Filesystem paths (relative to project root)
    public const MODEL_PATH = __DIR__ . '/../../models/model.rbx';
    public const REHEARSAL_POOL_PATH = __DIR__ . '/../../data/rehearsal_pool.json';
    public const REPLAY_BUFFER_PATH = __DIR__ . '/../../data/replay_buffer.json';
    public const GAMES_SINCE_REPLAY_PATH = __DIR__ . '/../../data/games_since_replay.txt';
}
