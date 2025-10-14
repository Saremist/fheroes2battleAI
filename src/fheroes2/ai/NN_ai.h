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
    class BattleMLP;

    const int INPUT_SIZE = 240; // Size of the input feature vector
    const int HIDDEN_SIZE = 128; // Size of the LSTM hidden state
    const int LAYER_NUM = 1; // Number of LSTM layers

    extern std::shared_ptr<NNAI::BattleMLP> g_model1;
    extern std::shared_ptr<NNAI::BattleMLP> g_model2;
    // Global model pointers for each color
    extern std::shared_ptr<BattleMLP> g_model_blue;
    extern std::shared_ptr<BattleMLP> g_model_green;
    extern std::shared_ptr<BattleMLP> g_model_red;
    extern std::shared_ptr<BattleMLP> g_model_yellow;
    extern std::shared_ptr<BattleMLP> g_model_orange;
    extern std::shared_ptr<BattleMLP> g_model_purple;

    extern std::vector<torch::Tensor> g_states1;
    extern std::vector<std::vector<torch::Tensor>> g_actions1;
    extern std::vector<torch::Tensor> g_rewards1;
    extern std::vector<torch::Tensor> g_states2;
    extern std::vector<std::vector<torch::Tensor>> g_actions2;
    extern std::vector<torch::Tensor> g_rewards2;

    extern int m1WinCount;
    extern int m2WinCount;

    extern bool isTraining; // Defines if post battle dialog will open or the training loop will continue
    extern bool skipDebugLog; // Defines if post battle dialog will open or the training loop will continue
    extern bool isComparing; // Defines if game is comparing NNAI with Original AI

    extern int prevEnemyHP1, prevAllyHP1, prevEnemyUnits1, prevAllyUnits1;
    extern int prevEnemyHP2, prevAllyHP2, prevEnemyUnits2, prevAllyUnits2;

    const int HeadCount = 5; // Number of output heads in the model

    extern torch::Device device;

    struct BattleMLPImpl : torch::nn::Module
    {
        // Simpler MLP backbone
        torch::nn::Linear fc1{ nullptr };
        torch::nn::Linear fc2{ nullptr };
        torch::nn::Linear fc3{ nullptr };

        // Output heads
        torch::nn::Linear action_type_head{ nullptr };
        torch::nn::Linear position_x_head{ nullptr };
        torch::nn::Linear position_y_head{ nullptr };
        torch::nn::Linear destination_x_head{ nullptr };
        torch::nn::Linear destination_y_head{ nullptr };

        BattleMLPImpl( int64_t input_size = INPUT_SIZE )
            : fc1( torch::nn::Linear( input_size, 256 ) )
            , fc2( torch::nn::Linear( 256, 128 ) )
            , fc3( torch::nn::Linear( 128, 64 ) )
            , action_type_head( 64, 4 )
            , position_x_head( 64, 9 )
            , position_y_head( 64, 11 )
            , destination_x_head( 64, 9 )
            , destination_y_head( 64, 11 )
        {
            register_module( "fc1", fc1 );
            register_module( "fc2", fc2 );
            register_module( "fc3", fc3 );
            register_module( "action_type_head", action_type_head );
            register_module( "position_x_head", position_x_head );
            register_module( "position_y_head", position_y_head );
            register_module( "destination_x_head", destination_x_head );
            register_module( "destination_y_head", destination_y_head );

            torch::nn::init::xavier_uniform_( fc1->weight );
            torch::nn::init::xavier_uniform_( fc2->weight );
            torch::nn::init::xavier_uniform_( fc3->weight );
        }

        std::vector<torch::Tensor> forward( torch::Tensor x )
        {
            // Expect input shape [batch, input_size]
            if ( x.dim() > 2 )
                x = x.view( { x.size( 0 ), -1 } ); // flatten sequence if needed

            auto out = torch::relu( fc1( x ) );
            out = torch::relu( fc2( out ) );
            out = torch::relu( fc3( out ) );

            auto action_type_logits = action_type_head( out );
            auto position_x_logits = position_x_head( out );
            auto position_y_logits = position_y_head( out );
            auto destination_x_logits = destination_x_head( out );
            auto destination_y_logits = destination_y_head( out );

            return { action_type_logits, position_x_logits, position_y_logits, destination_x_logits, destination_y_logits };
        }
    };

    TORCH_MODULE( BattleMLP );

    // Model management
    void initializeGlobalModels();
    void createAndSaveModel( const std::string & model_path );
    std::shared_ptr<BattleMLP> getModelByColor( int color );
    void saveModel( const BattleMLP & model, const std::string & model_path );
    void loadModel( std::shared_ptr<BattleMLP> & modelPtr, const std::string & model_path );
    // torch::Tensor preprocessInput( const std::vector<float> & raw_data );
    torch::Tensor prepareBattleMLPInput( const Battle::Arena & arena, const Battle::Unit & currentUnit );
    Battle::Actions planUnitTurn( Battle::Arena & arena, const Battle::Unit & currentUnit );

    // Returns two random models and their names.
    std::tuple<BattleMLP &, std::string, BattleMLP &, std::string, BattleMLP &, std::string> SelectRandomModels();

    void trainingGameLoop( bool isFirstGameRun, bool isProbablyDemoVersion );

    int training_main( int argc, char ** argv, int64_t num_epochs, double learning_rate, torch::Device device, int64_t NUM_SELF_PLAY_GAMES );

    bool isNNControlled( int color ); // TODO MW

    void tryTrainModel( BattleMLP & model, torch::optim::Optimizer & optimizer, const std::vector<torch::Tensor> & states,
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
