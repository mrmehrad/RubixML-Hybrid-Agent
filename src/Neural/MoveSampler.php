<?php

declare(strict_types=1);

namespace HybridAgent\Neural;

use HybridAgent\Core\Config;

/**
 * Move selection with controlled creativity (Boltzmann exploration).
 *
 * Only meant to be called once the neural net is already confident in a
 * move (i.e. past the confidence-threshold gate); creativity is only ever
 * applied among "safe" candidates, never as a substitute for the Minimax
 * decision made under uncertainty.
 */
class MoveSampler
{
    public function __construct(
        private readonly float $creativityChance = Config::CREATIVITY_CHANCE,
        private readonly float $creativityTemperature = Config::CREATIVITY_TEMPERATURE,
    ) {
    }

    public function sample(array $probabilities, array $availableMoves, int $bestMove): int
    {
        if ((mt_rand() / mt_getrandmax()) > $this->creativityChance) {
            return $bestMove;
        }

        $weights = [];
        foreach ($availableMoves as $m) {
            $p = max($probabilities[(string) $m] ?? 0.0001, 0.0001);
            $weights[$m] = exp(log($p) / $this->creativityTemperature);
        }

        $sum = array_sum($weights);
        $r = (mt_rand() / mt_getrandmax()) * $sum;
        $cumulative = 0.0;

        foreach ($weights as $move => $w) {
            $cumulative += $w;
            if ($r <= $cumulative) {
                return $move;
            }
        }

        return $bestMove;
    }
}
