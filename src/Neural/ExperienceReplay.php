<?php

declare(strict_types=1);

namespace HybridAgent\Neural;

use Rubix\ML\Datasets\Labeled;

/**
 * Experience Replay (learning from past mistakes).
 *
 * After every game against a human, each AI decision point is stored with
 * the "correct" Minimax answer (not necessarily whatever move the AI
 * actually played). That means even when the AI blunders, the stored
 * training sample holds the right answer — so partial training makes it
 * learn directly from its own mistakes.
 */
class ExperienceReplay
{
    private array $buffer = [];

    public function __construct(private readonly string $bufferPath)
    {
        if (file_exists($bufferPath)) {
            $this->buffer = json_decode(file_get_contents($bufferPath), true) ?? [];
        }
    }

    public function record(array $context, string $correctMove): void
    {
        $this->buffer[] = ['context' => $context, 'move' => $correctMove];
    }

    public function count(): int
    {
        return count($this->buffer);
    }

    public function save(): void
    {
        file_put_contents($this->bufferPath, json_encode($this->buffer));
    }

    public function clear(): void
    {
        $this->buffer = [];
    }

    /**
     * Build the final dataset for partial training: new experiences + a
     * random sample of old data (rehearsal) so the model doesn't forget
     * what it already knew (catastrophic forgetting).
     */
    public function buildTrainingBatch(array $rehearsalSamples, array $rehearsalLabels, int $rehearsalSize): Labeled
    {
        $samples = array_column($this->buffer, 'context');
        $labels = array_column($this->buffer, 'move');

        $poolSize = count($rehearsalSamples);
        $pickCount = min($rehearsalSize, $poolSize);

        if ($pickCount > 0) {
            $indices = $pickCount === $poolSize
                ? array_keys($rehearsalSamples)
                : (array) array_rand($rehearsalSamples, $pickCount);

            foreach ($indices as $i) {
                $samples[] = $rehearsalSamples[$i];
                $labels[] = $rehearsalLabels[$i];
            }
        }

        return new Labeled($samples, $labels);
    }
}
