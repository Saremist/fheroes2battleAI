#pragma once
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <torch/torch.h>

#include "NN_ai.h"

namespace NNAI
{

    struct GenomeStats
    {
        float fitness = 0.0f;
        int wins = 0;
        int games = 0;
    };

    class Neuroevolution
    {
    public:
        Neuroevolution( size_t populationSize = 20, float mutationSigma = 0.02f, float eliteFraction = 0.2f );

        // Initialize population
        void initializePopulation();

        // Evaluate models using your existing battle loop
        // Should call external battle function: playBattle(modelA, modelB) → float [0..1]
        void evaluatePopulation( std::function<float( BattleCNN &, BattleCNN & )> playBattle );

        // Evolve to next generation
        void evolve();

        // Save best model to file
        void saveBestModel( const std::string & path ) const;

        std::vector<std::shared_ptr<BattleCNN>> sampleOpponents( std::shared_ptr<BattleCNN> agent, int num_opponents );

        // Accessors
        const std::vector<std::shared_ptr<BattleCNN>> & getPopulation() const
        {
            return population;
        }
        const std::vector<GenomeStats> & getStats() const
        {
            return stats;
        }

    private:
        torch::Tensor flattenParameters( const BattleCNN & model ) const;
        void loadParameters( BattleCNN & model, const torch::Tensor & flat ) const;
        torch::Tensor mutate( const torch::Tensor & flat ) const;
        torch::Tensor crossover( const torch::Tensor & p1, const torch::Tensor & p2 ) const;

    private:
        size_t POP_SIZE;
        float SIGMA;
        float ELITE_FRAC;

        std::vector<std::shared_ptr<BattleCNN>> population;
        std::vector<GenomeStats> stats;
    };

} // namespace NNAI
