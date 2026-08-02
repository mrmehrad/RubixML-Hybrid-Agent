<?php

declare(strict_types=1);

namespace HybridAgent\Neural;

use HybridAgent\Core\Config;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\NeuralNet\ActivationFunctions\ReLU;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\PersistentModel;
use Rubix\ML\Serializers\Native;

/**
 * Owns the model lifecycle: build the network architecture, run the
 * initial full training pass, persist/reload from disk, and run partial
 * (rehearsal) training rounds triggered by Experience Replay.
 */
class Trainer
{
    private Filesystem $persister;
    private Native $serializer;

    public function __construct(private readonly string $modelPath = Config::MODEL_PATH)
    {
        $this->persister = new Filesystem($this->modelPath);
        $this->serializer = new Native();
    }

    public function modelExists(): bool
    {
        return file_exists($this->modelPath);
    }

    public function load(): PersistentModel
    {
        return PersistentModel::load($this->persister, $this->serializer);
    }

    public function trainNew(Labeled $dataset): PersistentModel
    {
        $baseEstimator = new MultilayerPerceptron([
            new Dense(128),
            new Activation(new ReLU()),
            new Dense(64),
            new Activation(new ReLU()),
        ], 64, new Adam(0.001), 1e-4, 500);

        $estimator = new PersistentModel($baseEstimator, $this->persister, $this->serializer);
        $estimator->train($dataset);

        return $estimator;
    }

    public function save(PersistentModel $estimator): void
    {
        $estimator->save();
    }

    public function partialTrain(PersistentModel $estimator, Labeled $trainingBatch): void
    {
        $estimator->partial($trainingBatch);
        $estimator->save();
    }
}
