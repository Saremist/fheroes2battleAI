#ifndef FHEROES2_AI_NN_AI_H
#define FHEROES2_AI_NN_AI_H

#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <ai_battle.h>
#include <battle_board.h>
#include <battle_troop.h>

#include <torch/torch.h>
#include "ostream"


namespace NNAI
{
    class BattleLSTM;
    extern std::shared_ptr<NNAI::BattleLSTM> g_model1;
    extern std::shared_ptr<NNAI::BattleLSTM> g_model2;
    extern const bool isTraining;
    extern const int TrainingLoopsCount;


    extern torch::Device device;

    struct BattleLSTMImpl : torch::nn::Module
    {
        torch::nn::LSTM lstm_layer{ nullptr };

        // Output heads
        torch::nn::Linear action_type_head{ nullptr }; // 6 types: SKIP, MOVE, ATTACK, SPELLCAST, RETREAT, SURRENDER
        torch::nn::Linear position_head{ nullptr }; // For MOVE, ATTACK, SPELLCAST (position index 0-98)
        torch::nn::Linear target_id_head{ nullptr }; // For ATTACK (up to 5 enemies)
        torch::nn::Linear direction_head{ nullptr }; // For ATTACK (0-6 directions)

        BattleLSTMImpl( int64_t input_size = 15, int64_t hidden_size = 128, int64_t num_layers = 1 )
            : lstm_layer( torch::nn::LSTMOptions( input_size, hidden_size ).num_layers( num_layers ).batch_first( true ) )
            , action_type_head( hidden_size, 6 )
            , position_head( hidden_size, 99 )
            , target_id_head( hidden_size, 5 )
            , direction_head( hidden_size, 6 )
        {
            register_module( "lstm_layer", lstm_layer );
            register_module( "action_type_head", action_type_head );
            register_module( "position_head", position_head );
            register_module( "target_id_head", target_id_head );
            register_module( "direction_head", direction_head );

                        // --- Initialize LSTM ---
            for ( int layer = 0; layer < num_layers; ++layer ) {
                torch::NoGradGuard no_grad;
                // Use named_parameters() to access by string key
                auto params = lstm_layer->named_parameters();
                auto w_ih = params.find( "weight_ih_l" + std::to_string( layer ) );
                auto w_hh = params.find( "weight_hh_l" + std::to_string( layer ) );
                auto b_ih = params.find( "bias_ih_l" + std::to_string( layer ) );
                auto b_hh = params.find( "bias_hh_l" + std::to_string( layer ) );

                if ( w_hh != nullptr )
                    torch::nn::init::orthogonal_( *w_hh );
                if ( w_ih != nullptr )
                    torch::nn::init::xavier_uniform_( *w_ih );
                if ( b_ih != nullptr )
                    b_ih->zero_();
                if ( b_hh != nullptr )
                    b_hh->zero_();

                // Forget gate bias to 1.0
                int64_t hidden = hidden_size;
                if ( b_ih != nullptr )
                    b_ih->slice( 0, hidden, 2 * hidden ).fill_( 1.0 );
                if ( b_hh != nullptr )
                    b_hh->slice( 0, hidden, 2 * hidden ).fill_( 1.0 );
            }

            // --- Initialize output heads (Xavier) ---
            torch::nn::init::xavier_uniform_( action_type_head->weight );
            torch::nn::init::xavier_uniform_( position_head->weight );
            torch::nn::init::xavier_uniform_( target_id_head->weight );
            torch::nn::init::xavier_uniform_( direction_head->weight );

            torch::nn::init::constant_( action_type_head->bias, 0 );
            torch::nn::init::constant_( position_head->bias, 0 );
            torch::nn::init::constant_( target_id_head->bias, 0 );
            torch::nn::init::constant_( direction_head->bias, 0 );
        }


        std::vector<torch::Tensor> forward( torch::Tensor x )
        {
            auto h0 = torch::zeros( { lstm_layer->options.num_layers(), x.size( 0 ), lstm_layer->options.hidden_size() } );
            auto c0 = torch::zeros( { lstm_layer->options.num_layers(), x.size( 0 ), lstm_layer->options.hidden_size() } );

            auto lstm_out = std::get<0>( lstm_layer( x, std::make_tuple( h0, c0 ) ) );

            // Take output from last time step
            auto last_timestep = lstm_out.select( 1, lstm_out.size( 1 ) - 1 ); // Shape: [batch, hidden]

            // Output multiple heads
            torch::Tensor action_type_logits = action_type_head( last_timestep );
            torch::Tensor position_logits = position_head( last_timestep );
            torch::Tensor target_id_logits = target_id_head( last_timestep );
            torch::Tensor direction_logits = direction_head( last_timestep );

            return { action_type_logits, position_logits, target_id_logits, direction_logits };
        }
    };

    TORCH_MODULE( BattleLSTM );

    // Model management
    void initializeGlobalModels();
    void createAndSaveModel( const std::string & model_path );
    std::shared_ptr<BattleLSTM> getModelByColor( int color );
    void saveModel( const BattleLSTM & model, const std::string & model_path );
    void loadModel( std::shared_ptr<BattleLSTM>& modelPtr, const std::string & model_path );
    // Using Models
    void trainModel( BattleLSTM & model, int64_t num_epochs, double learning_rate, torch::Device device, int64_t NUM_SELF_PLAY_GAMES );
    std::vector<torch::Tensor> predict( BattleLSTM & model, const torch::Tensor & input );
    //torch::Tensor preprocessInput( const std::vector<float> & raw_data );
    torch::Tensor prepareBattleLSTMInput( const Battle::Arena & arena, const Battle::Unit & currentUnit );
    Battle::Actions predict_action(const Battle::Unit & currentUnit, Battle::Arena & arena );
    Battle::Actions planUnitTurn( Battle::Arena & arena, const Battle::Unit & currentUnit );

    void trainingGameLoop( bool isFirstGameRun, bool isProbablyDemoVersion, int training_loops );

    bool isNNControlled( int color ); // TODO MW
}

void PrintUnitInfo( const Battle::Unit & unit );

#include "ostream"
#include <string>
#include "battle_command.h"

namespace Battle
{
    const char * CommandTypeToString( CommandType type );
    std::ostream & operator<<( std::ostream & os, const Command & command );
    std::ostream & operator<<( std::ostream & os, const Actions & actions );
}

#endif // FHEROES2_AI_NN_AI_H
