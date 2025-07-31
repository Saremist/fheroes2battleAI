#ifndef FHEROES2_AI_NN_AI_H
#define FHEROES2_AI_NN_AI_H

#pragma once

#pragma warning( disable : 4996 )

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
    // Global model pointers for each color
    extern std::shared_ptr<BattleLSTM> g_model_blue;
    extern std::shared_ptr<BattleLSTM> g_model_green;
    extern std::shared_ptr<BattleLSTM> g_model_red;
    extern std::shared_ptr<BattleLSTM> g_model_yellow;
    extern std::shared_ptr<BattleLSTM> g_model_orange;
    extern std::shared_ptr<BattleLSTM> g_model_purple;

    extern std::vector<torch::Tensor> * g_states1;
    extern std::vector<std::vector<torch::Tensor>> * g_actions1;
    extern std::vector<torch::Tensor> * g_rewards1;
    extern std::vector<torch::Tensor> * g_states2;
    extern std::vector<std::vector<torch::Tensor>> * g_actions2;
    extern std::vector<torch::Tensor> * g_rewards2;

    extern int m1skipCount;
    extern int m2skipCount;
    extern int m1CorrectMovesCount;
    extern int m2CorrectMovesCount;
    extern int m1turnCount;
    extern int m2turnCount;

    extern bool isTraining; // Defines if post battle dialog will open or the training loop will continue
    extern bool skipDebugLog; // Defines if post battle dialog will open or the training loop will continue
    extern const int TrainingLoopsCount;

    extern int prevEnemyHP1, prevAllyHP1, prevEnemyUnits1, prevAllyUnits1;
    extern int prevEnemyHP2, prevAllyHP2, prevEnemyUnits2, prevAllyUnits2;

    extern torch::Device device;

    struct BattleLSTMImpl : torch::nn::Module
    {
        torch::nn::LSTM lstm_layer{ nullptr };

        // Output heads
        torch::nn::Linear action_type_head{ nullptr }; // 4 types: SKIP, MOVE, ATTACK, SPELLCAST
        torch::nn::Linear position_head{ nullptr }; // For MOVE, ATTACK, SPELLCAST (position index 0-98)
        torch::nn::Linear direction_head{ nullptr }; // For ATTACK (0-6 directions)

        BattleLSTMImpl( int64_t input_size = 17, int64_t hidden_size = 128, int64_t num_layers = 1 )
            : lstm_layer( torch::nn::LSTMOptions( input_size, hidden_size ).num_layers( num_layers ).batch_first( true ) )
            , action_type_head( hidden_size, 4 )
            , position_head( hidden_size, 99 )
            , direction_head( hidden_size, 7 )
        {
            register_module( "lstm_layer", lstm_layer );
            register_module( "action_type_head", action_type_head );
            register_module( "position_head", position_head );
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
            torch::nn::init::xavier_uniform_( direction_head->weight );

            torch::nn::init::constant_( action_type_head->bias, 0 );
            torch::nn::init::constant_( position_head->bias, 0 );
            torch::nn::init::constant_( direction_head->bias, 0 );
        }

        std::vector<torch::Tensor> forward( torch::Tensor x )
        {
            auto h0 = torch::zeros( { lstm_layer->options.num_layers(), x.size( 0 ), lstm_layer->options.hidden_size() }, x.options() ).to( x.device() );
            auto c0 = torch::zeros( { lstm_layer->options.num_layers(), x.size( 0 ), lstm_layer->options.hidden_size() }, x.options() ).to( x.device() );

            auto lstm_out = std::get<0>( lstm_layer( x, std::make_tuple( h0, c0 ) ) ).to( x.device() );

            // Take output from last time step
            auto last_timestep = lstm_out.select( 1, lstm_out.size( 1 ) - 1 ); // Shape: [batch, hidden]

            // Output multiple heads
            torch::Tensor action_type_logits = action_type_head( last_timestep );
            torch::Tensor position_logits = position_head( last_timestep );
            torch::Tensor direction_logits = direction_head( last_timestep );

            return { action_type_logits, position_logits, direction_logits };
        }
    };

    TORCH_MODULE( BattleLSTM );

    // Model management
    void initializeGlobalModels();
    void createAndSaveModel( const std::string & model_path );
    std::shared_ptr<BattleLSTM> getModelByColor( int color );
    void saveModel( const BattleLSTM & model, const std::string & model_path );
    void loadModel( std::shared_ptr<BattleLSTM> & modelPtr, const std::string & model_path );
    // torch::Tensor preprocessInput( const std::vector<float> & raw_data );
    torch::Tensor prepareBattleLSTMInput( const Battle::Arena & arena, const Battle::Unit & currentUnit );
    Battle::Actions planUnitTurn( Battle::Arena & arena, const Battle::Unit & currentUnit );

    // Returns two random models and their names.
    std::tuple<BattleLSTM &, std::string, BattleLSTM &, std::string> SelectRandomModels();

    void trainingGameLoop( bool isFirstGameRun, bool isProbablyDemoVersion );

    int training_main( int argc, char ** argv, int64_t num_epochs, double learning_rate, torch::Device device, int64_t NUM_SELF_PLAY_GAMES );

    bool isNNControlled( int color ); // TODO MW

    void tryTrainModel( BattleLSTM & model, torch::optim::Optimizer & optimizer, const std::vector<torch::Tensor> & states,
                        const std::vector<std::vector<torch::Tensor>> & actions, const std::vector<torch::Tensor> & rewards, float & total_loss,
                        float & epoch_total_reward, torch::Device device, int model_id, int game_index );
    void resetGameRewardStats( Battle::Arena & arena );
} // NNAI

void PrintUnitInfo( const Battle::Unit & unit );

#include <string>

#include "battle_command.h"
#include "ostream"

namespace Battle
{
    const char * CommandTypeToString( CommandType type );
    std::ostream & operator<<( std::ostream & os, const Command & command );
    std::ostream & operator<<( std::ostream & os, const Actions & actions );
    float calculateReward( const Battle::Arena & currArena, int color );
}

#endif // FHEROES2_AI_NN_AI_H
