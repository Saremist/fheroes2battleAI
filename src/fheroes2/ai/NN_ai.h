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

#include "battle_command.h"
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

    extern int m1WinCount;
    extern int m2WinCount;

    extern bool isTraining; // Defines if post battle dialog will open or the training loop will continue
    extern bool skipDebugLog; // Defines if post battle dialog will open or the training loop will continue
    extern bool isComparing; // Defines if game is comparing NNAI with Original AI

    extern int prevEnemyHP1, prevAllyHP1, prevEnemyUnits1, prevAllyUnits1;
    extern int prevEnemyHP2, prevAllyHP2, prevEnemyUnits2, prevAllyUnits2;

    const int HeadCount = 5; // Number of output heads in the model

    extern torch::Device device;

    struct BattleLSTMImpl : torch::nn::Module
    {
        torch::nn::LSTM lstm_layer{ nullptr };

        // Output heads
        torch::nn::Linear action_type_head{ nullptr }; // 4 types: SKIP, MOVE, ATTACK, SPELLCAST
        torch::nn::Linear position_x_head{ nullptr }; // For MOVE, ATTACK, SPELLCAST (x coordinate)
        torch::nn::Linear position_y_head{ nullptr }; // For MOVE, ATTACK, SPELLCAST (y coordinate)
        // torch::nn::Linear direction_head{ nullptr }; // For MOVE, ATTACK, SPELLCAST (y coordinate)
        torch::nn::Linear destination_x_head{ nullptr }; // For ATTACK (x coordinate)
        torch::nn::Linear destination_y_head{ nullptr }; // For ATTACK (y coordinate)

        BattleLSTMImpl( int64_t input_size = 24, int64_t hidden_size = 128, int64_t num_layers = 1 )
            : lstm_layer( torch::nn::LSTMOptions( input_size, hidden_size ).num_layers( num_layers ).batch_first( true ) )
            , action_type_head( hidden_size, 4 )
            , position_x_head( hidden_size, 9 )
            , position_y_head( hidden_size, 11 )
            //, direction_head( hidden_size, 7 )
            , destination_x_head( hidden_size, 9 )
            , destination_y_head( hidden_size, 11 )
        {
            register_module( "lstm_layer", lstm_layer );
            register_module( "action_type_head", action_type_head );
            register_module( "position_x_head", position_x_head );
            register_module( "position_y_head", position_y_head );
            // register_module( "direction_head", direction_head );
            register_module( "destination_x_head", destination_x_head );
            register_module( "destination_y_head", destination_y_head );

            // --- Initialize LSTM ---
            for ( int layer = 0; layer < num_layers; ++layer ) {
                torch::NoGradGuard no_grad;
                for ( auto & param : lstm_layer->named_parameters() ) {
                    if ( param.key().find( "weight" ) != std::string::npos ) {
                        if ( param.key().find( "ih" ) != std::string::npos ) {
                            torch::nn::init::xavier_uniform_( param.value() );
                        }
                        else if ( param.key().find( "hh" ) != std::string::npos ) {
                            torch::nn::init::orthogonal_( param.value() );
                        }
                    }
                    else if ( param.key().find( "bias" ) != std::string::npos ) {
                        param.value().zero_();
                        // Forget gate bias to 1.0
                        int64_t hidden = hidden_size;
                        param.value().slice( 0, hidden, 2 * hidden ).fill_( 1.0 );
                    }
                }
            }

            // --- Initialize output heads (Xavier) ---
            torch::nn::init::xavier_uniform_( action_type_head->weight );
            torch::nn::init::xavier_uniform_( position_x_head->weight );
            torch::nn::init::xavier_uniform_( position_y_head->weight );
            // torch::nn::init::xavier_uniform_( direction_head->weight );
            torch::nn::init::xavier_uniform_( destination_x_head->weight );
            torch::nn::init::xavier_uniform_( destination_y_head->weight );

            torch::nn::init::constant_( action_type_head->bias, 0 );
            torch::nn::init::constant_( position_x_head->bias, 0 );
            torch::nn::init::constant_( position_y_head->bias, 0 );
            // torch::nn::init::constant_( direction_head->bias, 0 );
            torch::nn::init::constant_( destination_x_head->bias, 0 );
            torch::nn::init::constant_( destination_y_head->bias, 0 );
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
            torch::Tensor position_x_logits = position_x_head( last_timestep );
            torch::Tensor position_y_logits = position_y_head( last_timestep );
            // torch::Tensor direction_logits = direction_head( last_timestep );
            torch::Tensor destination_x_logits = destination_x_head( last_timestep );
            torch::Tensor destination_y_logits = destination_y_head( last_timestep );

            // return { action_type_logits, position_x_logits, position_y_logits, direction_logits };
            return { action_type_logits, position_x_logits, position_y_logits, destination_x_logits, destination_y_logits };
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
    std::tuple<BattleLSTM &, std::string, BattleLSTM &, std::string, BattleLSTM &, std::string> SelectRandomModels();

    void trainingGameLoop( bool isFirstGameRun, bool isProbablyDemoVersion );

    int training_main( int argc, char ** argv, int64_t num_epochs, double learning_rate, torch::Device device, int64_t NUM_SELF_PLAY_GAMES );

    bool isNNControlled( int color ); // TODO MW

    void tryTrainModel( BattleLSTM & model, torch::optim::Optimizer & optimizer, const std::vector<torch::Tensor> & states,
                        const std::vector<std::vector<torch::Tensor>> & actions, const std::vector<torch::Tensor> & rewards, float & total_loss,
                        float & epoch_total_reward, torch::Device device, int model_id );
    void resetGameRewardStats( Battle::Arena & arena );

    inline std::pair<int, int> getXYCoordinates( const Battle::Unit & unit )
    {
        // ( unit.GetHeadIndex() / Board::widthInCells ) + 1 ) + ", " + std::to_string( ( unit.GetHeadIndex() % Board::widthInCells ) + 1 )
        int x = ( unit.GetHeadIndex() / Battle::Board::widthInCells );
        int y = ( unit.GetHeadIndex() % Battle::Board::widthInCells );
        return { x, y };
    }

    inline int getIndexFromXY( int x, int y )
    {
        return ( x * Battle::Board::widthInCells ) + y;
    }

    inline float normalize( float value, float min, float max )
    {
        return ( static_cast<float>( value ) - static_cast<float>( min ) ) / ( static_cast<float>( max ) - static_cast<float>( min ) );
    }
} // NNAI

void PrintUnitInfo( const Battle::Unit & unit );

namespace Battle
{
    const char * CommandTypeToString( CommandType type );
    std::ostream & operator<<( std::ostream & os, const Command & command );
    std::ostream & operator<<( std::ostream & os, const Actions & actions );
    float calculateReward( const Battle::Arena & currArena, int color );
}

#endif // FHEROES2_AI_NN_AI_H
