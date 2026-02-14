#ifndef FHEROES2_AI_NN_AI_H
#define FHEROES2_AI_NN_AI_H

#pragma once

#pragma warning( disable : 4996 )

#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <random>
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

    // ---- Configuration constants ----
    const int INPUT_SIZE = 7 * 10; // Size of the input feature vector feature count * troop count
    const int ACTION_SIZE = 99 + 1; // Number of discrete actions tiels on board + skip
    const int HIDDEN_SIZE = 128 * 4; // Hidden layer size for the Q-network
    const int NUM_HIDDEN_LAYERS = 2; // Number of hidden layers in the MLP Q-network
    const size_t REPLAY_BUFFER_CAPACITY = 10000; // Experience replay capacity
    const double GAMMA = 0.99; // Discount factor
    const double TAU = 1e-3; // For soft update of target network (if used)
    const double EPS_START = 1.0; // Initial epsilon for epsilon-greedy
    const double EPS_END = 0.05; // Minimum epsilon
    // const double EPS_DECAY = 1e-5; // Epsilon decay per step (or use multiplicative decay)

    extern int blue_monster_count;
    extern int red_monster_count;

    extern torch::Device device;

    extern torch::Tensor initial_game_state_blue;
    extern torch::Tensor initial_game_state_red;
    extern int saved_action;

    extern int enemyType;

    // ---- Simple MLP Q-network ----
    struct QNetworkImpl : torch::nn::Module
    {
        torch::nn::Linear fc1{ nullptr };
        torch::nn::ModuleList hidden_layers{ nullptr };
        torch::nn::Linear fc_out{ nullptr };

        QNetworkImpl( int64_t input_dim = INPUT_SIZE, int64_t hidden_dim = HIDDEN_SIZE, int64_t output_dim = ACTION_SIZE, int num_hidden = NUM_HIDDEN_LAYERS )
            : fc1( input_dim, hidden_dim )
            , hidden_layers( torch::nn::ModuleList() )
            , fc_out( hidden_dim, output_dim )
        {
            register_module( "fc1", fc1 );

            // Create hidden layers safely inside ModuleList
            for ( int i = 0; i < num_hidden - 1; ++i ) {
                auto layer = torch::nn::Linear( hidden_dim, hidden_dim );
                hidden_layers->push_back( layer );
            }
            register_module( "hidden_layers", hidden_layers );
            register_module( "fc_out", fc_out );

            // Initialize weights
            torch::nn::init::xavier_uniform_( fc1->weight );
            torch::nn::init::constant_( fc1->bias, 0 );

            for ( auto & m : *hidden_layers ) {
                if ( auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>( m ) ) {
                    torch::nn::init::xavier_uniform_( linear->weight );
                    torch::nn::init::constant_( linear->bias, 0 );
                }
            }

            torch::nn::init::xavier_uniform_( fc_out->weight );
            torch::nn::init::constant_( fc_out->bias, 0 );
        }

        torch::Tensor forward( torch::Tensor x )
        {
            x = torch::relu( fc1->forward( x ) );
            for ( auto & m : *hidden_layers ) {
                if ( auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>( m ) ) {
                    x = torch::relu( linear->forward( x ) );
                }
            }
            return fc_out->forward( x );
        }
    };

    TORCH_MODULE( QNetwork );

    // ---- Experience tuple and Replay Buffer ----
    struct Experience
    {
        torch::Tensor state; // shape: [INPUT_SIZE] or [1, INPUT_SIZE]
        torch::Tensor prev_state; // shape: [INPUT_SIZE] or [1, INPUT_SIZE]
        int64_t action; // discrete action index
        double reward; // scalar
        bool done; // terminal flag
    };

    class ReplayBuffer
    {
    public:
        ReplayBuffer( size_t capacity = REPLAY_BUFFER_CAPACITY );

        void push( const Experience & exp );
        std::vector<Experience> sample( size_t batch_size );
        std::vector<Experience> get_all() const; // ?? add this
        size_t size() const noexcept;

        double get_last_reward() const;
        bool set_last_reward( double reward );
        bool ReplayBuffer::propagate_rewards_back( double reward, std::size_t n );
        void clear();

    private:
        std::deque<Experience> buffer_;
        size_t capacity_;
        std::mt19937 rng_;
    };

    // ---- Per-color per type models & shared training state ----
    extern std::shared_ptr<QNetwork> g_qmodel_blue;
    extern std::shared_ptr<QNetwork> g_qmodel_red;
    // extern std::shared_ptr<QNetwork> g_qmodel_blue_ranged;
    // extern std::shared_ptr<QNetwork> g_qmodel_red_ranged;

    // Optionally a target network per color (for stability)
    extern std::shared_ptr<QNetwork> g_target_blue;
    extern std::shared_ptr<QNetwork> g_target_red;
    // extern std::shared_ptr<QNetwork> g_target_blue_ranged;
    // extern std::shared_ptr<QNetwork> g_target_red_ranged;

    // per-model buffers (choose one approach)
    extern std::shared_ptr<ReplayBuffer> g_replay_buffer_blue;
    extern std::shared_ptr<ReplayBuffer> g_replay_buffer_red;
    // extern std::shared_ptr<ReplayBuffer> g_replay_buffer_blue_ranged;
    // extern std::shared_ptr<ReplayBuffer> g_replay_buffer_red_ranged;

    // Training state
    extern bool isTraining;
    extern bool skipDebugLog;
    extern bool isRunningExperiments;

    extern int episodesPerSeries;

    extern bool StateInitialized;

    extern double epsilon; // Current epsilon for epsilon-greedy
    extern int64_t training_steps_done;

    // ---- Utility / API ----

    // Model lifecycle
    void initialize_qmodels( torch::Device dev = torch::kCPU );
    void create_and_save_qmodel( const std::string & model_path ); // creates a fresh model and saves to path
    void save_qmodel( const QNetwork & model, const std::string & model_path );
    void load_qmodel( std::shared_ptr<QNetwork> & modelPtr, const std::string & model_path );

    std::shared_ptr<QNetwork> getQModelByColorAndType( int color, bool isRanged );

    // Epsilon-greedy action selection
    // state_tensor must be a 1D tensor shape [INPUT_SIZE] or 2D [1,INPUT_SIZE]
    int selectActionEpsilonGreedy( std::shared_ptr<QNetwork> model, torch::Tensor state_tensor );

    // Deterministic choice (argmax)
    int selectActionGreedy( std::shared_ptr<QNetwork> model, torch::Tensor state_tensor );

    // Convert arena & unit -> state tensor
    torch::Tensor prepareStateTensor( const Battle::Arena & arena, const Battle::Unit & currentUnit );

    // Convert an action index back to in-game Actions / Command
    Battle::Actions actionIndexToGameActions( int action_index, Battle::Arena & arena, const Battle::Unit & currentUnit );

    Battle::Actions AttackClosestEnemy( Battle::Arena & arena, const Battle::Unit & currentUnit );

    Battle::Actions DefendClosestAlly( Battle::Arena & arena, const Battle::Unit & currentUnit );

    std::vector<int> selectTopActions( std::shared_ptr<QNetwork> model, const torch::Tensor & state );

    // Add experience to replay buffer

    // Optimize the model with a batch sampled from replay buffer
    // optimizer provided externally to keep flexibility
    void optimize_model( QNetwork & model, torch::optim::Optimizer & optimizer, std::shared_ptr<ReplayBuffer> replay_buffer, double gamma, torch::Device device );

    // Soft update target network parameters: target = tau*local + (1-tau)*target
    void soft_update_target( QNetwork & local_model, QNetwork & target_model, double tau );

    bool isNNControlled( int color );

    int training_main( int argc, char ** argv, int64_t num_series, double learning_rate, torch::Device device, int64_t episodes_per_series );
    void trainingGameLoop( bool isFirstGameRun, bool isProbablyDemoVersion );

    void remember_experience( const torch::Tensor & state, const torch::Tensor & next_state, int64_t action, double reward, bool done, int color, bool isRanged );

    // Battle::Actions planUnitTurn( Battle::Arena & arena, const Battle::Unit & currentUnit );

    Battle::Actions NeuralPlanTurn( Battle::Arena & arena, const Battle::Unit & currentUnit );

    Battle::Actions AgresivePlanTurn( Battle::Arena & arena, const Battle::Unit & currentUnit );

    Battle::Actions RandomPlanTurn( Battle::Arena & arena, const Battle::Unit & currentUnit );

    int getClosestNeighborIndex( const Battle::Unit & unit, const int32_t targetIndex, Battle::Arena & arena );

    // Quick helpers
    inline int getIndexFromXY( int x, int y )
    {
        return ( x * Battle::Board::widthInCells ) + y;
    }

    inline std::pair<int, int> getXYCoordinates( const Battle::Unit & unit )
    {
        int x = ( unit.GetHeadIndex() / Battle::Board::widthInCells );
        int y = ( unit.GetHeadIndex() % Battle::Board::widthInCells );
        return { x, y };
    }

    inline float normalize( float value, float min, float max )
    {
        if ( max == min )
            return 0.0f;
        return ( static_cast<float>( value ) - static_cast<float>( min ) ) / ( static_cast<float>( max ) - static_cast<float>( min ) );
    }

} // namespace NNAI

// Non-member helpers (battle-related)
void PrintUnitInfo( const Battle::Unit & unit );

namespace Battle
{
    const char * CommandTypeToString( CommandType type );
    std::ostream & operator<<( std::ostream & os, const Command & command );
    std::ostream & operator<<( std::ostream & os, const Actions & actions );
    float calculateReward( const torch::Tensor & prev_state, const torch::Tensor & curr_state, int color );
}

#endif // FHEROES2_AI_NN_AI_H
