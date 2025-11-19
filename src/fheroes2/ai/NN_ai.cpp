#include "NN_ai.h"

#pragma warning( disable : 4996 )

#include <algorithm>
#include <deque>
#include <filesystem>
#include <iostream>
#include <random>
#include <tuple>

#include <torch/torch.h>

#include "battle.h"
#include "battle_arena.h"
#include "battle_army.h"
#include "battle_command.h"
#include "game.h"
#include "ui_tool.h"

namespace NNAI
{
    // --- Global Q-networks (one per color) ---
    std::shared_ptr<QNetwork> g_qmodel_blue = nullptr;
    std::shared_ptr<QNetwork> g_qmodel_red = nullptr;

    std::shared_ptr<QNetwork> g_target_blue = nullptr;
    std::shared_ptr<QNetwork> g_target_red = nullptr;

    std::shared_ptr<ReplayBuffer> g_replay_buffer_blue = nullptr;
    std::shared_ptr<ReplayBuffer> g_replay_buffer_red = nullptr;

    bool isTraining = true;
    bool skipDebugLog = true;
    bool isComparing = true;

    torch::Tensor saved_game_state = torch::Tensor();
    int saved_action = 0;

    double epsilon = EPS_START;
    int64_t training_steps_done = 0;

    torch::Device device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU );

    // --- ReplayBuffer implementation ---
    ReplayBuffer::ReplayBuffer( size_t capacity )
        : capacity_( capacity )
        , rng_( std::random_device{}() )
    {}

    void ReplayBuffer::push( const Experience & exp )
    {
        if ( buffer_.size() >= capacity_ )
            buffer_.pop_front();
        buffer_.push_back( exp );
    }

    std::vector<Experience> ReplayBuffer::sample( size_t batch_size )
    {
        std::vector<Experience> batch;
        batch.reserve( batch_size );

        if ( batch_size == 0 || buffer_.empty() )
            return batch;

        std::uniform_int_distribution<size_t> dist( 0, buffer_.size() - 1 );
        for ( size_t i = 0; i < batch_size; ++i ) {
            batch.push_back( buffer_[dist( rng_ )] );
        }
        return batch;
    }

    size_t ReplayBuffer::size() const noexcept
    {
        return buffer_.size();
    }

    void ReplayBuffer::clear()
    {
        buffer_.clear();
    }

    // --- Model lifecycle helpers ---
    void create_and_save_qmodel( const std::string & model_path )
    {
        try {
            QNetwork model( INPUT_SIZE, HIDDEN_SIZE, ACTION_SIZE, NUM_HIDDEN_LAYERS );
            model->to( device );
            namespace fs = std::filesystem;
            // fs::create_directories( fs::path( model_path ).parent_path() );
            torch::save( model, model_path );
            if ( !skipDebugLog )
                std::cout << "Created and saved new Q-model to " << model_path << std::endl;
        }
        catch ( const std::exception & e ) {
            std::cerr << "Error creating or saving Q-model: " << e.what() << std::endl;
        }
    }

    void save_qmodel( const QNetwork & model, const std::string & model_path )
    {
        try {
            torch::save( model, model_path );
            if ( !skipDebugLog )
                std::cout << "Q-model saved to " << model_path << std::endl;
        }
        catch ( const std::exception & e ) {
            std::cerr << "Error saving Q-model: " << e.what() << std::endl;
        }
    }

    void load_qmodel( std::shared_ptr<QNetwork> & modelPtr, const std::string & model_path )
    {
        namespace fs = std::filesystem;
        try {
            if ( !fs::exists( model_path ) ) {
                if ( !skipDebugLog )
                    std::cerr << "Q-model file does not exist at " << model_path << ". Creating default model..." << std::endl;
                create_and_save_qmodel( model_path );
            }
            modelPtr = std::make_shared<QNetwork>( INPUT_SIZE, HIDDEN_SIZE, ACTION_SIZE, NUM_HIDDEN_LAYERS );
            torch::load( *modelPtr, model_path );
            modelPtr->get()->to( device );
            if ( !skipDebugLog )
                std::cout << "Loaded Q-model from " << model_path << std::endl;
        }
        catch ( const std::exception & e ) {
            std::cerr << "Error loading Q-model: " << e.what() << std::endl;
            modelPtr = nullptr;
        }
    }

    // Initialize global models and replay buffer
    void initialize_qmodels( torch::Device dev )
    {
        device = dev;
        g_replay_buffer_blue = std::make_shared<ReplayBuffer>( REPLAY_BUFFER_CAPACITY );
        g_replay_buffer_red = std::make_shared<ReplayBuffer>( REPLAY_BUFFER_CAPACITY );

        load_qmodel( g_qmodel_blue, "qmodel_blue.pt" );
        load_qmodel( g_qmodel_red, "qmodel_red.pt" );

        if ( g_qmodel_blue ) {
            g_target_blue = std::make_shared<QNetwork>( *g_qmodel_blue );
            g_target_blue->get()->to( device );
        }
        if ( g_qmodel_red ) {
            g_target_red = std::make_shared<QNetwork>( *g_qmodel_red );
            g_target_red->get()->to( device );
        }
    }

    // --- Action selection ---
    int selectActionGreedy( std::shared_ptr<QNetwork> model, torch::Tensor state_tensor )
    {
        if ( !model )
            return 0;
        // Ensure shape [1, INPUT_SIZE]
        if ( state_tensor.dim() == 1 )
            state_tensor = state_tensor.unsqueeze( 0 );
        state_tensor = state_tensor.to( device ).to( torch::kFloat32 );

        model->get()->eval();
        torch::NoGradGuard no_grad;
        auto qvals = model->get()->forward( state_tensor ); // shape [1, ACTION_SIZE]
        auto action = std::get<1>( qvals.max( 1 ) ).item<int64_t>();
        return static_cast<int>( action );
    }

    int selectActionEpsilonGreedy( std::shared_ptr<QNetwork> model, torch::Tensor state_tensor )
    {
        // epsilon-greedy using global epsilon
        std::uniform_real_distribution<double> dist( 0.0, 1.0 );
        static std::mt19937 rng( std::random_device{}() );
        double sample = dist( rng );
        if ( sample < epsilon || !model ) {
            // random action
            std::uniform_int_distribution<int> a_dist( 0, ACTION_SIZE - 1 );
            return a_dist( rng );
        }
        else {
            return selectActionGreedy( model, state_tensor );
        }
    }

    std::shared_ptr<QNetwork> getQModelByColor( int color )
    {
        switch ( color ) {
        case 0x01: // BLUE
            return g_qmodel_blue;
        case 0x04: // RED
            return g_qmodel_red;
        default:
            std::cerr << "Warning: Unrecognized color " << color << ". Returning default model." << std::endl;
            return nullptr;
        }
    }

    // --- State preprocessing ---
    // Re-use and adapt the old feature extraction but produce a flat tensor of length INPUT_SIZE
    std::vector<float> extractUnitFeaturesVec( const Battle::Unit & unit, const Battle::Arena & arena, const Battle::Unit & currentunit )
    {
        // Same features as before
        std::vector<float> features;
        std::pair<int, int> coords = getXYCoordinates( unit );

        features.push_back( static_cast<float>( unit.GetUID() ) );
        features.push_back( normalize( static_cast<float>( coords.first ), 0, 9 ) );
        features.push_back( normalize( static_cast<float>( coords.second ), 0, 11 ) );
        features.push_back( normalize( static_cast<float>( unit.GetCount() ), 0, 300 ) );
        features.push_back( normalize( static_cast<float>( unit.GetHitPoints() ), 0, 500 ) );
        features.push_back( normalize( static_cast<float>( unit.GetSpeed( false, true ) ), 0, 10 ) );
        features.push_back( normalize( static_cast<float>( arena.GetBoard()->GetDistance( currentunit.GetPosition(), unit.GetPosition() ) ), 0, 50 ) );
        features.push_back( normalize( static_cast<float>( unit.GetAttack() ), 0, 100 ) );
        features.push_back( normalize( static_cast<float>( unit.GetDefense() ), 0, 100 ) );
        features.push_back( unit.isFlying() ? 1.0f : 0.0f );
        features.push_back( unit.isArchers() ? 1.0f : 0.0f );
        features.push_back( normalize( static_cast<float>( unit.GetShots() ), 0, 50 ) );
        features.push_back( unit.isHandFighting() ? 1.0f : 0.0f );
        features.push_back( currentunit.GetColor() == unit.GetColor() ? 1.0f : 0.0f );
        features.push_back( unit.GetColor() == arena.GetArmy1Color() ? 1.0f : 0.0f );
        features.push_back( unit.Modes( Battle::TR_MOVED ) ? 1.0f : 0.0f );
        features.push_back( unit.Modes( Battle::TR_RESPONDED ) ? 1.0f : 0.0f );
        features.push_back(
            arena.GetBoard()->CanAttackTargetFromPosition( currentunit, unit, arena.GetBoard()->GetDistance( currentunit.GetPosition(), unit.GetPosition() ) ) ? 1.0f
                                                                                                                                                               : 0.0f );

        return features;
    }

    torch::Tensor prepareStateTensor( const Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        const Battle::Units enemies( arena.getEnemyForce( arena.GetCurrentColor() ).getUnits(), Battle::Units::REMOVE_INVALID_UNITS_AND_SPECIFIED_UNIT, &currentUnit );
        const Battle::Units allies( arena.GetCurrentForce().getUnits(), Battle::Units::REMOVE_INVALID_UNITS_AND_SPECIFIED_UNIT, &currentUnit );

        std::vector<std::vector<float>> featList;

        // Current unit
        featList.push_back( extractUnitFeaturesVec( currentUnit, arena, currentUnit ) );

        // Allies up to 4
        int allyCount = 0;
        for ( const Battle::Unit * u : allies ) {
            if ( u && u->isValid() && u != &currentUnit ) {
                featList.push_back( extractUnitFeaturesVec( *u, arena, currentUnit ) );
                if ( ++allyCount == 4 )
                    break;
            }
        }
        while ( allyCount < 4 ) {
            featList.push_back( std::vector<float>( featList[0].size(), 0.0f ) );
            ++allyCount;
        }

        // Enemies up to 5
        int enemyCount = 0;
        for ( const Battle::Unit * u : enemies ) {
            if ( u && u->isValid() ) {
                featList.push_back( extractUnitFeaturesVec( *u, arena, currentUnit ) );
                if ( ++enemyCount == 5 )
                    break;
            }
        }
        while ( enemyCount < 5 ) {
            featList.push_back( std::vector<float>( featList[0].size(), 0.0f ) );
            ++enemyCount;
        }

        // Flatten
        std::vector<float> flat;
        for ( auto & vec : featList ) {
            flat.insert( flat.end(), vec.begin(), vec.end() );
        }

        // Pad/truncate to INPUT_SIZE
        if ( flat.size() < static_cast<size_t>( INPUT_SIZE ) ) {
            flat.resize( INPUT_SIZE, 0.0f );
        }
        else if ( flat.size() > static_cast<size_t>( INPUT_SIZE ) ) {
            flat.resize( INPUT_SIZE );
        }

        torch::Tensor t = torch::from_blob( flat.data(), { INPUT_SIZE }, torch::kFloat32 ).clone().to( device );
        return t; // shape [INPUT_SIZE]
    }

    void trainingGameLoop( bool /*isFirstGameRun*/, bool /*isProbablyDemoVersion*/ )
    {
        fheroes2::GameMode result = fheroes2::GameMode::NEW_BATTLE_ONLY;

        bool exit = false;

        while ( !exit ) {
            switch ( result ) {
            case fheroes2::GameMode::QUIT_GAME:
                exit = true;
                break;
            case fheroes2::GameMode::MAIN_MENU:
                // result = Game::NewBattleOnly(); //loop back to battle only if trying to leave
                result = Game::MainMenu( false );
                break;
            case fheroes2::GameMode::NEW_GAME:
                result = Game::NewBattleOnly(); // new game sets up battle only
                break;
            case fheroes2::GameMode::NEW_BATTLE_ONLY:
                result = Game::NewBattleOnly();
                break;
            case fheroes2::GameMode::NEW_MULTI:
                result = Game::NewHotSeat();
                break;
            default:
                // If this assertion blows up then you are entering an infinite loop!
                // Add the logic for the newly added entry.
                assert( 0 );
                exit = true;
                break;
            }
        }
    }

    Battle::Actions planUnitTurn( Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        // If the unit already moved this turn, do nothing
        if ( currentUnit.Modes( Battle::TR_MOVED ) ) {
            return {};
        }

        // Choose model by color
        std::shared_ptr<QNetwork> model = getQModelByColor( currentUnit.GetColor() );

        // If no NN model available, fallback to SKIP to avoid crashes.
        if ( !model ) {
            if ( !skipDebugLog ) {
                std::cerr << "planUnitTurn: no Q-model for color " << currentUnit.GetColor() << " — SKIP\n";
            }
            Battle::Actions actions;
            actions.emplace_back( Battle::Command::SKIP, static_cast<int>( currentUnit.GetUID() ) );
            std::cout << "FORCING SKIP ERRORR!!!" << std::endl;
            return actions;
        }

        // Prepare state tensor for the current unit
        torch::Tensor state = prepareStateTensor( arena, currentUnit ); // shape [INPUT_SIZE]
        NNAI::saved_game_state = state;

        // Select action (epsilon-greedy). This function handles device/shape internally.
        int action_index = selectActionEpsilonGreedy( model, state );

        // Map discrete action index back to in-game actions
        Battle::Actions actions = actionIndexToGameActions( action_index, arena, currentUnit );

        // Optional debug output
        if ( !skipDebugLog ) {
            std::cout << "planUnitTurn: color=" << static_cast<int>( currentUnit.GetColor() ) << " uid=" << currentUnit.GetUID() << " action_index=" << action_index
                      << " -> " << actions << std::endl;
        }

        NNAI::saved_action = action_index;

        return actions;
    }

    // --- Memory API ---
    void remember_experience( const torch::Tensor & state, int64_t action, double reward, const torch::Tensor & next_state, bool done, int color )
    {
        if ( !g_replay_buffer_blue || !g_replay_buffer_red )
            return;
        Experience e;
        e.state = state.detach().to( torch::kCPU );
        e.action = action;
        e.reward = reward;
        e.next_state = next_state.detach().to( torch::kCPU );
        e.done = done;
        if ( color == 0x01 ) // BLUE
            g_replay_buffer_blue->push( e );
        else if ( color == 0x04 ) // RED
            g_replay_buffer_red->push( e );
    }

    // --- Optimization step ---
    void optimize_model( QNetwork & model, torch::optim::Optimizer & optimizer, std::shared_ptr<ReplayBuffer> replay_buffer, size_t batch_size, double gamma,
                         torch::Device device, float & out_loss )
    {
        if ( !replay_buffer )
            return;
        if ( replay_buffer->size() < batch_size )
            return;

        auto batch = replay_buffer->sample( batch_size );
        if ( batch.empty() )
            return;

        std::vector<torch::Tensor> states, next_states;
        std::vector<int64_t> actions;
        std::vector<float> rewards;
        std::vector<uint8_t> dones;

        for ( const auto & e : batch ) {
            states.push_back( e.state.to( device ) );
            next_states.push_back( e.next_state.to( device ) );
            actions.push_back( e.action );
            rewards.push_back( static_cast<float>( e.reward ) );
            dones.push_back( e.done ? 1u : 0u );
        }

        auto state_batch = torch::stack( states ); // [B, INPUT_SIZE]
        auto next_state_batch = torch::stack( next_states ); // [B, INPUT_SIZE]
        auto action_batch = torch::tensor( actions, torch::TensorOptions().dtype( torch::kLong ).device( device ) ); // [B]
        auto reward_batch = torch::tensor( rewards, torch::TensorOptions().dtype( torch::kFloat32 ).device( device ) ); // [B]
        auto done_batch = torch::tensor( dones, torch::TensorOptions().dtype( torch::kFloat32 ).device( device ) ); // [B]

        model->train();
        optimizer.zero_grad();

        // Current Q-values
        auto q_values_all = model->forward( state_batch ); // [B, ACTION_SIZE]
        auto q_values = q_values_all.gather( 1, action_batch.unsqueeze( 1 ) ).squeeze( 1 ); // [B]

        // compute max of next state's Q-values (bootstrap)
        auto next_q_values_all = model->forward( next_state_batch ); // [B, ACTION_SIZE]
        auto next_max_q = std::get<0>( next_q_values_all.max( 1 ) ); // [B]

        auto expected_q = reward_batch + ( 1.0 - done_batch ) * static_cast<float>( gamma ) * next_max_q;
        // MSE loss
        auto loss = torch::nn::functional::mse_loss( q_values, expected_q.detach() );

        loss.backward();
        optimizer.step();

        out_loss += loss.item<float>();
    }

    void NNAI::soft_update_target( QNetwork & local_model, QNetwork & target_model, double tau )
    {
        // θ_target = τ*θ_local + (1-τ)*θ_target
        torch::NoGradGuard no_grad;

        auto local_params = local_model->named_parameters();
        auto target_params = target_model->named_parameters();

        for ( auto & item : local_params ) {
            const auto & name = item.key();
            auto & local_tensor = item.value();
            auto & target_tensor = target_params[name];

            // new_val = (tau * local) + ((1 - tau) * target)
            torch::Tensor new_val = target_tensor.mul( 1.0 - tau ) + local_tensor.mul( tau );
            target_tensor.copy_( new_val );
        }
    }

    Battle::Actions actionIndexToGameActions( int action_index, Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        Battle::Actions actions;
        int uid = static_cast<int>( currentUnit.GetUID() );

        if ( action_index <= 0 ) {
            actions.emplace_back( Battle::Command::SKIP, uid );
            return actions;
        }

        int targetIndex = action_index - 1;
        // Clamp to valid board indices
        const int maxIndex = Battle::Board::widthInCells * Battle::Board::heightInCells - 1; // board dimension assumed
        if ( targetIndex < 0 )
            targetIndex = 0;
        if ( targetIndex > maxIndex )
            targetIndex = maxIndex;

        // If unit exists at target -> ATTACK
        const auto * cell = arena.GetBoard()->GetCell( targetIndex );
        int targetUnitUID = -1;
        if ( cell ) {
            const auto * unit = cell->GetUnit();
            if ( unit )
                targetUnitUID = unit->GetUID();
        }

        int positionNum = currentUnit.GetPosition().GetHead()->GetIndex();
        int attackDirection = -1;
        // If there is a target unit, compute direction
        if ( targetUnitUID != -1 ) {
            int attackTargetPosition = targetIndex;
            attackDirection = Battle::Board::GetDirection( positionNum, attackTargetPosition );
            // Validate attack parameters quickly
            if ( CheckAttackParameters( &currentUnit, ( cell ? cell->GetUnit() : nullptr ), positionNum, attackTargetPosition, attackDirection ) ) {
                actions.emplace_back( Battle::Command::ATTACK, uid, targetUnitUID, positionNum, attackTargetPosition, attackDirection );
                return actions;
            }
            // If invalid attack, fallthrough to attempt move
        }

        // Attempt move to the target cell (validate)
        if ( CheckMoveParameters( &currentUnit, targetIndex ) ) {
            actions.emplace_back( Battle::Command::MOVE, uid, targetIndex );
            return actions;
        }

        // As fallback SKIP
        actions.emplace_back( Battle::Command::SKIP, uid );
        return actions;
    }

    // --- Battle-specific helpers retained from original file (reward, print, etc.) ---

    // Previous state tracking for reward calculation
    int prevEnemyHP1 = -1, prevAllyHP1 = -1, prevEnemyUnits1 = -1, prevAllyUnits1 = -1;
    int prevEnemyHP2 = -1, prevAllyHP2 = -1, prevEnemyUnits2 = -1, prevAllyUnits2 = -1;

    void resetGameRewardStats( Battle::Arena & arena )
    {
        int color = arena.GetArmy1Color();
        prevEnemyHP1 = arena.getEnemyForce( color ).GetAliveHitPoints();
        prevAllyHP1 = arena.getForce( color ).GetAliveHitPoints();
        prevEnemyUnits1 = arena.getEnemyForce( color ).GetAliveCounts();
        prevAllyUnits1 = arena.getForce( color ).GetAliveCounts();

        color = arena.GetArmy2Color();
        prevEnemyHP2 = arena.getEnemyForce( color ).GetAliveHitPoints();
        prevAllyHP2 = arena.getForce( color ).GetAliveHitPoints();
        prevEnemyUnits2 = arena.getEnemyForce( color ).GetAliveCounts();
        prevAllyUnits2 = arena.getForce( color ).GetAliveCounts();
    }

    bool isNNControlled( int color )
    {
        if ( isComparing ) {
            switch ( color ) {
            case 0x01: // BLUE
                return true;
            case 0x04: // RED
                return false;
            }
        }
        return true;
    }

} // namespace NNAI

// --- Non-namespace helpers kept mostly as in your original file ---
void PrintUnitInfo( const Battle::Unit & unit )
{
    std::cout << "Unit Name: " << unit.GetName() << ", Unit Id:" << unit.GetID() << ", Unit UID: " << unit.GetUID() << ", Count: " << unit.GetCount()
              << ", Position: " << unit.GetPosition().GetHead()->GetIndex() << ", Shooting: " << unit.GetShots() << ", speed: " << unit.GetSpeed()
              << " HitPoints: " << unit.GetHitPointsLeft() << std::endl;
}

#include <string>

#include "ostream"

namespace Battle
{
    const char * CommandTypeToString( CommandType type )
    {
        switch ( type ) {
        case CommandType::MOVE:
            return "MOVE";
        case CommandType::ATTACK:
            return "ATTACK";
        case CommandType::SPELLCAST:
            return "SPELLCAST";
        case CommandType::MORALE:
            return "MORALE";
        case CommandType::CATAPULT:
            return "CATAPULT";
        case CommandType::TOWER:
            return "TOWER";
        case CommandType::RETREAT:
            return "RETREAT";
        case CommandType::SURRENDER:
            return "SURRENDER";
        case CommandType::SKIP:
            return "SKIP";
        case CommandType::AUTO_SWITCH:
            return "AUTO_SWITCH";
        case CommandType::AUTO_FINISH:
            return "AUTO_FINISH";
        default:
            return "UNKNOWN";
        }
    }

    std::ostream & operator<<( std::ostream & os, const Command & command )
    {
        os << "Command(" << CommandTypeToString( command.GetType() ) << ") [";

        bool first = true;
        for ( const int param : command ) {
            if ( !first )
                os << ", ";
            os << param;
            first = false;
        }

        os << "]";
        return os;
    }

    std::ostream & operator<<( std::ostream & os, const Actions & actions )
    {
        os << "Actions [\n";
        for ( const Command & cmd : actions ) {
            os << "  " << cmd << "\n";
        }
        os << "]";
        return os;
    }

    float calculateReward( const torch::Tensor & prev_state, const torch::Tensor & curr_state, int color )
    {
        // Config constants matching prepareStateTensor / extractUnitFeaturesVec
        const int total_slots = 10; // 1 current + 4 allies + 5 enemies
        const float HP_MAX = 500.0f; // must match normalization used in extractUnitFeaturesVec

        // Validate shapes
        if ( !prev_state.defined() || !curr_state.defined() || prev_state.numel() == 0 || curr_state.numel() == 0 ) {
            // If no previous state available, we cannot compute delta — return 0 (but still check win later)
            // Check for win condition via curr_state -> if enemy HP all zero, give win reward
            try {
                // compute enemy total hp from curr_state below
            }
            catch ( ... ) {
                return 0.0f;
            }
        }

        // Work on CPU and contiguous
        torch::Tensor prev = prev_state.detach().cpu().contiguous();
        torch::Tensor cur = curr_state.detach().cpu().contiguous();

        // Determine feature_size: INPUT_SIZE must be divisible by total_slots
        int64_t input_elems = prev.numel();
        if ( input_elems != static_cast<int64_t>( NNAI::INPUT_SIZE ) ) {
            // If the incoming tensor shape doesn't match INPUT_SIZE fall back to safe behavior:
            // try to reshape or clamp, but here we'll attempt best-effort: if divisible by total_slots use that
            if ( input_elems % total_slots != 0 ) {
                // Can't interpret layout reliably
                if ( !NNAI::skipDebugLog )
                    std::cerr << "[calculateRewardFromStates] Unexpected prev_state size " << input_elems << " (expected " << NNAI::INPUT_SIZE << ")\n";
                return 0.0f;
            }
        }
        int feature_size = static_cast<int>( input_elems / total_slots );

        // Enemy slots are at indices 5..9 (0=current, 1-4 allies)
        const int enemy_slot_start = 1 + 4; // =5
        const int enemy_slot_count = 5;

        // Hitpoints index inside per-unit features (based on extractUnitFeaturesVec)
        const int hitpoint_idx = 4;

        // Sum normalized hitpoints across enemy slots for prev and cur
        float prev_enemy_hp_norm_sum = 0.0f;
        float curr_enemy_hp_norm_sum = 0.0f;

        for ( int slot = 0; slot < enemy_slot_count; ++slot ) {
            int base = ( enemy_slot_start + slot ) * feature_size;
            int hp_idx = base + hitpoint_idx;
            if ( hp_idx < 0 || hp_idx >= prev.numel() )
                continue; // safety

            float prev_hp_norm = prev[hp_idx].item<float>();
            float cur_hp_norm = cur[hp_idx].item<float>();

            // clamp to [0,1] as the extractor should produce normalized values
            if ( !std::isfinite( prev_hp_norm ) )
                prev_hp_norm = 0.0f;
            if ( !std::isfinite( cur_hp_norm ) )
                cur_hp_norm = 0.0f;
            prev_hp_norm = std::max( 0.0f, std::min( 1.0f, prev_hp_norm ) );
            cur_hp_norm = std::max( 0.0f, std::min( 1.0f, cur_hp_norm ) );

            prev_enemy_hp_norm_sum += prev_hp_norm;
            curr_enemy_hp_norm_sum += cur_hp_norm;
        }

        // Convert normalized sums to raw HP sums (approximate using HP_MAX)
        float prev_enemy_hp_sum = prev_enemy_hp_norm_sum * HP_MAX;
        float curr_enemy_hp_sum = curr_enemy_hp_norm_sum * HP_MAX;

        // Compute delta (damage dealt)
        float delta_enemy_hp = prev_enemy_hp_sum - curr_enemy_hp_sum;

        // Reward: percent of damage relative to previous enemy HP sum (avoid division by zero)
        float reward = 0.0f;
        float denom = ( prev_enemy_hp_sum > 1e-6f ) ? prev_enemy_hp_sum : 1.0f;
        if ( delta_enemy_hp > 0.0f ) {
            reward += 100.0f * ( delta_enemy_hp / denom );
        }

        // Win bonus: if estimated enemy HP is zero (all enemy hp slots are zero)
        if ( curr_enemy_hp_sum <= 0.0f + 1e-6f ) {
            reward += 1000.0f;
            if ( !NNAI::skipDebugLog )
                std::cout << "[DEBUG] calculateRewardFromStates: Win detected (enemy HP sum zero)." << std::endl;
        }

        // Ensure non-negative
        reward = std::max( reward, 0.0f );

        if ( !NNAI::skipDebugLog ) {
            std::cout << "[DEBUG] calculateRewardFromStates: prev_hp_sum=" << prev_enemy_hp_sum << ", curr_hp_sum=" << curr_enemy_hp_sum << ", delta=" << delta_enemy_hp
                      << ", reward=" << reward << std::endl;
        }

        return reward;
    }
} // namespace Battle
