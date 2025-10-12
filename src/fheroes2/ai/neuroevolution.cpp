#include "neuroevolution.h"

#include <iostream>

#include "NN_ai.h"

using namespace NNAI;

Neuroevolution::Neuroevolution( size_t populationSize, float mutationSigma, float eliteFraction )
    : POP_SIZE( populationSize )
    , SIGMA( mutationSigma )
    , ELITE_FRAC( eliteFraction )
{
    initializePopulation();
}

void Neuroevolution::initializePopulation()
{
    population.clear();
    stats.resize( POP_SIZE );
    for ( size_t i = 0; i < POP_SIZE; ++i )
        population.push_back( std::make_shared<BattleCNN>() );
}

torch::Tensor Neuroevolution::flattenParameters( const BattleCNN & model ) const
{
    std::vector<torch::Tensor> params;
    for ( auto & p : model->parameters() )
        params.push_back( p.view( -1 ) );
    return torch::cat( params );
}

void Neuroevolution::loadParameters( BattleCNN & model, const torch::Tensor & flat ) const
{
    int64_t offset = 0;
    for ( auto & p : model->parameters() ) {
        auto numel = p.numel();
        auto slice = flat.slice( 0, offset, offset + numel ).view_as( p );
        p.data().copy_( slice );
        offset += numel;
    }
}

torch::Tensor Neuroevolution::mutate( const torch::Tensor & flat ) const
{
    return flat + SIGMA * torch::randn_like( flat );
}

torch::Tensor Neuroevolution::crossover( const torch::Tensor & p1, const torch::Tensor & p2 ) const
{
    torch::Tensor mask = ( torch::rand_like( p1 ) > 0.5 );
    return mask * p1 + ( ~mask ) * p2;
}

void Neuroevolution::evaluatePopulation( std::function<float( BattleCNN &, BattleCNN & )> playBattle )
{
    for ( auto & s : stats ) {
        s.fitness = 0;
        s.wins = 0;
        s.games = 0;
    }

    // round-robin evaluation
    for ( size_t i = 0; i < population.size(); ++i ) {
        for ( size_t j = i + 1; j < population.size(); ++j ) {
            float result = playBattle( *population[i], *population[j] ); // 1.0 if i wins, 0.0 if j wins
            stats[i].fitness += result;
            stats[j].fitness += ( 1.0f - result );
            stats[i].games++;
            stats[j].games++;
            if ( result > 0.5f )
                stats[i].wins++;
            else
                stats[j].wins++;
        }
    }

    for ( auto & s : stats )
        if ( s.games > 0 )
            s.fitness /= s.games; // normalize to [0,1]
}

void Neuroevolution::evolve()
{
    // Sort indices by fitness
    std::vector<size_t> idx( POP_SIZE );
    std::iota( idx.begin(), idx.end(), 0 );
    std::sort( idx.begin(), idx.end(), [&]( size_t a, size_t b ) { return stats[a].fitness > stats[b].fitness; } );

    size_t eliteCount = std::max<size_t>( 1, static_cast<size_t>( POP_SIZE * ELITE_FRAC ) );

    std::vector<std::shared_ptr<BattleCNN>> newPop;
    newPop.reserve( POP_SIZE );

    // Keep elites
    for ( size_t i = 0; i < eliteCount; ++i )
        newPop.push_back( population[idx[i]] );

    // Create offspring
    while ( newPop.size() < POP_SIZE ) {
        size_t pa = idx[rand() % eliteCount];
        size_t pb = idx[rand() % eliteCount];
        auto child = std::make_shared<BattleCNN>();
        auto g1 = flattenParameters( *population[pa] );
        auto g2 = flattenParameters( *population[pb] );
        auto childGenome = mutate( crossover( g1, g2 ) );
        loadParameters( *child, childGenome );
        newPop.push_back( child );
    }

    population = std::move( newPop );
    stats.resize( POP_SIZE );

    std::cout << "[NeuroEvo] Evolved new generation. Best fitness: " << stats[idx[0]].fitness << std::endl;
}

std::vector<std::shared_ptr<BattleCNN>> Neuroevolution::sampleOpponents( std::shared_ptr<BattleCNN> agent, int num_opponents )
{
    std::vector<std::shared_ptr<BattleCNN>> selected;
    std::vector<size_t> indices( population.size() );
    std::iota( indices.begin(), indices.end(), 0 );

    // Remove agent itself
    size_t agent_idx = 0;
    for ( size_t i = 0; i < population.size(); ++i ) {
        if ( population[i] == agent ) { // Compare shared_ptrs directly
            agent_idx = i;
            break;
        }
    }
    indices.erase( indices.begin() + agent_idx );

    std::shuffle( indices.begin(), indices.end(), std::mt19937{ std::random_device{}() } );

    for ( int i = 0; i < num_opponents && i < (int)indices.size(); ++i ) {
        selected.push_back( population[indices[i]] );
    }

    return selected;
}

void Neuroevolution::saveBestModel( const std::string & path ) const
{
    size_t bestIdx = std::distance( stats.begin(), std::max_element( stats.begin(), stats.end(),
                                                                     []( const GenomeStats & a, const GenomeStats & b ) { return a.fitness < b.fitness; } ) );

    try {
        torch::save( *population[bestIdx], path );
        std::cout << "[NeuroEvo] Saved best model to " << path << std::endl;
    }
    catch ( const c10::Error & e ) {
        std::cerr << "[NeuroEvo] Failed to save best model: " << e.msg() << std::endl;
    }
}
