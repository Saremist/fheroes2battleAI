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
    // ---- Per-color per type models & shared training state ----
    extern std::shared_ptr<QNetwork> g_qmodel_blue = nullptr;
    extern std::shared_ptr<QNetwork> g_qmodel_red = nullptr;
    extern std::shared_ptr<QNetwork> g_qmodel_blue_ranged = nullptr;
    extern std::shared_ptr<QNetwork> g_qmodel_red_ranged = nullptr;

    // Optionally a target network per color (for stability )
    extern std::shared_ptr<QNetwork> g_target_blue = nullptr;
    extern std::shared_ptr<QNetwork> g_target_red = nullptr;
    extern std::shared_ptr<QNetwork> g_target_blue_ranged = nullptr;
    extern std::shared_ptr<QNetwork> g_target_red_ranged = nullptr;

    // per-model buffers (choose one approach)
    extern std::shared_ptr<ReplayBuffer> g_replay_buffer_blue = nullptr;
    extern std::shared_ptr<ReplayBuffer> g_replay_buffer_red = nullptr;
    extern std::shared_ptr<ReplayBuffer> g_replay_buffer_blue_ranged = nullptr;
    extern std::shared_ptr<ReplayBuffer> g_replay_buffer_red_ranged = nullptr;

    bool isTraining = true;
    bool skipDebugLog = true;
    bool isRunningExperiments = true;
    int episodesPerSeries = 500;

    bool StateInitialized = false;

    int blue_monster_count = 1;
    int red_monster_count = 5;

    int enemyType = -1;

    torch::Tensor initial_game_state_blue = torch::Tensor();
    torch::Tensor initial_game_state_red = torch::Tensor();
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

    bool ReplayBuffer::propagate_rewards_back( double reward, std::size_t n )
    {
        if ( buffer_.empty() )
            return false;

        n = std::min( n, buffer_.size() );

        for ( std::size_t i = 1; i < n; ++i ) {
            buffer_[buffer_.size() - 1 - i].reward = reward / ( i + 1 );
        }

        return true;
    }

    bool ReplayBuffer::set_last_reward( double reward )
    {
        if ( buffer_.empty() )
            return false;

        buffer_.back().reward = reward;
        return true;
    }

    double ReplayBuffer::get_last_reward() const
    {
        if ( buffer_.empty() )
            return 0.0;
        return buffer_.back().reward;
    }

    std::vector<Experience> ReplayBuffer::get_all() const
    {
        return std::vector<Experience>( buffer_.begin(), buffer_.end() );
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
        // g_replay_buffer_blue_ranged = std::make_shared<ReplayBuffer>( REPLAY_BUFFER_CAPACITY );
        // g_replay_buffer_red_ranged = std::make_shared<ReplayBuffer>( REPLAY_BUFFER_CAPACITY );

        load_qmodel( g_qmodel_blue, "qmodel_blue.pt" );
        load_qmodel( g_qmodel_red, "qmodel_red.pt" );
        // load_qmodel( g_qmodel_blue_ranged, "qmodel_blue_ranged.pt" );
        // load_qmodel( g_qmodel_red_ranged, "qmodel_red_ranged.pt" );

        if ( g_qmodel_blue ) {
            g_target_blue = std::make_shared<QNetwork>( *g_qmodel_blue );
            g_target_blue->get()->to( device );
        }
        if ( g_qmodel_red ) {
            g_target_red = std::make_shared<QNetwork>( *g_qmodel_red );
            g_target_red->get()->to( device );
        }
        /*       if ( g_qmodel_blue_ranged ) {
                   g_target_blue_ranged = std::make_shared<QNetwork>( *g_qmodel_blue_ranged );
                   g_target_blue_ranged->get()->to( device );
               }
               if ( g_qmodel_red_ranged ) {
                   g_target_red_ranged = std::make_shared<QNetwork>( *g_qmodel_red_ranged );
                   g_target_red_ranged->get()->to( device );
               }*/
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

        // Q-values: [1, ACTION_SIZE]
        torch::Tensor qvals = model->get()->forward( state_tensor ).squeeze( 0 );

        float max_q = qvals.max().item<float>();
        torch::Tensor best_actions = torch::nonzero( qvals == max_q ).squeeze( 1 );

        // Randomly pick one of them
        int num_best = best_actions.size( 0 );

        if ( num_best == 1 )
            return best_actions.item<int>();

        static thread_local std::mt19937 rng{ std::random_device{}() };
        std::uniform_int_distribution<int> dist( 0, num_best - 1 );

        int chosen = best_actions[dist( rng )].item<int>();
        return chosen;
    }

    std::shared_ptr<QNetwork> getQModelByColorAndType( int color, bool isRanged )
    {
        switch ( color ) {
        case 0x01: // BLUE
                   // if ( isRanged )
                   //    return g_qmodel_blue_ranged;
                   // else
            return g_qmodel_blue;
        case 0x04: // RED
                   // if ( isRanged )
                   //    return g_qmodel_red_ranged;
                   // else
            return g_qmodel_red;

        default:
            std::cerr << "Warning: Unrecognized color " << color << ". Returning default model." << std::endl;
            return nullptr;
        }
    }

    // --- State preprocessing ---
    std::vector<float> extractUnitFeaturesVec( const Battle::Unit & unit, const Battle::Arena & arena, const Battle::Unit & currentunit )
    {
        int currentUnitTotalHits = currentunit.GetHitPoints() * currentunit.GetCount();

        // Same features as before
        std::vector<float> features;
        std::pair<int, int> coords = getXYCoordinates( unit );

        features.push_back( unit.GetHitPoints() * unit.GetCount() );

        features.push_back( static_cast<float>( coords.first ) );
        features.push_back( static_cast<float>( coords.second ) );

        features.push_back( normalize( unit.GetHitPoints() * unit.GetCount(), 0, currentUnitTotalHits ) );
        features.push_back( normalize( unit.GetSpeed( false, true ), 0, currentunit.GetSpeed( false, true ) ) );
        features.push_back( normalize( arena.GetBoard()->GetDistance( currentunit.GetPosition(), unit.GetPosition() ), 0, currentunit.GetSpeed( false, true ) ) );
        features.push_back( normalize( unit.GetAttack(), 0, currentunit.GetAttack() ) );
        features.push_back( normalize( unit.GetDefense(), 0, currentunit.GetDefense() ) );
        features.push_back( unit.isArchers() ? 1.0f : 0.0f );
        features.push_back( unit.GetColor() == arena.GetArmy1Color() ? 1.0f : 0.0f );

        // std::cout << features << std::endl;

        return features;
    }

    torch::Tensor prepareStateTensor( const Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        const Battle::Units allies( arena.GetCurrentForce().getUnits(), Battle::Units::REMOVE_INVALID_UNITS_AND_SPECIFIED_UNIT, &currentUnit );
        const Battle::Units enemies( arena.getEnemyForce( arena.GetCurrentColor() ).getUnits(), Battle::Units::REMOVE_INVALID_UNITS_AND_SPECIFIED_UNIT, &currentUnit );

        std::vector<std::vector<float>> featList;

        // Current unit
        if ( !currentUnit.isValid() ) {
            // std::cout << "AAAAAAAAAAAA" << std::endl; // TODO
            return torch::zeros( { INPUT_SIZE }, torch::kFloat32 ).to( device );
        }
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

    Battle::Actions NeuralPlanTurn( Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        // Choose model by color
        std::shared_ptr<QNetwork> model = getQModelByColorAndType( currentUnit.GetColor(), bool( currentUnit.GetShots() > 0 ) );

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

        std::vector<int> candidates = selectTopActions( model, state );

        for ( int action_index : candidates ) {
            Battle::Actions acts = actionIndexToGameActions( action_index, arena, currentUnit );

            // If the mapping didn't turn into SKIP, return it
            if ( !acts.empty() && acts.front().GetType() != Battle::Command::SKIP ) {
                NNAI::saved_action = action_index;
                return acts;
            }
        }

        // No legal action found → SKIP
        Battle::Actions a;
        a.emplace_back( Battle::Command::SKIP, currentUnit.GetUID() );
        return a;
    }

    Battle::Actions AgresivePlanTurn( Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        return NeuralPlanTurn( arena, currentUnit );
    }

    Battle::Actions RandomPlanTurn( Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        Battle::Actions actions;
        while ( true ) {
            int targetCellIndex = rand() % 99;

            // std::cout << "Random target Cell: " << targetCellIndex << std::endl;

            const auto * targetCell = arena.GetBoard()->GetCell( targetCellIndex );

            uint32_t uid = currentUnit.GetUID();
            int targetUnitUID = -1;
            if ( targetCell ) {
                const auto * unit = targetCell->GetUnit();
                if ( unit )
                    targetUnitUID = unit->GetUID();
            }

            if ( currentUnit.GetPosition().GetHead()->GetIndex() == targetCellIndex ) { // If already on target cell, attempt SKIP
                actions.emplace_back( Battle::Command::SKIP, uid );
                return actions;
            }
            else if ( targetUnitUID != -1 ) { // Do archery first
                int positionNum = -1;
                int attackDirection = -1;
                if ( currentUnit.GetShots() <= 0 ) { // Swap to  mele if archery not available
                    positionNum = getClosestNeighborIndex( currentUnit, targetCellIndex, arena );
                    attackDirection = Battle::Board::GetDirection( positionNum, targetCellIndex );
                }

                // Validate attack parameters quickly
                if ( CheckAttackParameters( &currentUnit, ( targetCell ? targetCell->GetUnit() : nullptr ), positionNum, targetCellIndex, attackDirection ) ) {
                    actions.emplace_back( Battle::Command::ATTACK, uid, targetUnitUID, positionNum, targetCellIndex, attackDirection );
                    return actions;
                }
                // If invalid attack, fallthrough to attempt move
            }
            else if ( CheckMoveParameters( &currentUnit, targetCellIndex ) ) { // Attempt move to the target cell (validate)
                actions.emplace_back( Battle::Command::MOVE, uid, targetCellIndex );
                return actions;
            }
            /*else {
                actions.emplace_back( Battle::Command::SKIP, uid );
                std::cout << "FORCED SKIP" << std::endl;
                return actions;
            }*/
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

        int targetIndex = action_index;
        // Clamp to valid board indices
        const int maxIndex = Battle::Board::widthInCells * Battle::Board::heightInCells - 1;
        if ( targetIndex < 0 )
            targetIndex = 0;
        if ( targetIndex > maxIndex ) {
            int actionIndex = targetIndex - maxIndex - 1;

            if ( actionIndex == 0 ) {
                actions.splice( actions.end(), AttackClosestEnemy( arena, currentUnit ) );
                return actions;
            }
            else if ( actionIndex == 1 ) {
                actions.splice( actions.end(), DefendClosestAlly( arena, currentUnit ) );
                return actions;
            }
            else {
                actions.emplace_back( Battle::Command::SKIP, uid );
            }

            /*int enemyIndex = targetIndex - maxIndex - 1;
            const Battle::Units enemies( arena.getEnemyForce( arena.GetCurrentColor() ).getUnits(), Battle::Units::REMOVE_INVALID_UNITS_AND_SPECIFIED_UNIT,
                                         &currentUnit );M
            if ( enemies.size() > enemyIndex ) {
                targetIndex = enemies[enemyIndex]->GetHeadIndex();
            }
            else {
                actions.emplace_back( Battle::Command::SKIP, uid );
            }*/
        }

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
            positionNum = getClosestNeighborIndex( currentUnit, attackTargetPosition, arena );
            attackDirection = Battle::Board::GetDirection( positionNum, attackTargetPosition );
            if ( currentUnit.GetShots() > 0 ) {
                attackDirection = -1; // swap to archery if available
                positionNum = -1;
            }
            // Validate attack parameters quickly
            if ( CheckAttackParameters( &currentUnit, ( cell ? cell->GetUnit() : nullptr ), positionNum, attackTargetPosition, attackDirection ) ) {
                actions.emplace_back( Battle::Command::ATTACK, uid, targetUnitUID, positionNum, attackTargetPosition, attackDirection );
                return actions;
            }
            // If invalid attack, fallthrough to attempt move
        }

        // Attempt move to the target cell (validate)
        if ( CheckMoveParameters( &currentUnit, targetIndex ) ) {
            actions.emplace_back( Battle::Command::MOVE, currentUnit.GetUID(), targetIndex );
            return actions;
        }

        // As fallback SKIP
        actions.emplace_back( Battle::Command::SKIP, uid );
        return actions;
    }

    Battle::Actions AttackClosestEnemy( Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        Battle::Actions actions;
        int uid = static_cast<int>( currentUnit.GetUID() );
        const Battle::Units enemies( arena.getEnemyForce( arena.GetCurrentColor() ).getUnits(), Battle::Units::REMOVE_INVALID_UNITS_AND_SPECIFIED_UNIT, &currentUnit );
        const Battle::Unit * closestEnemy = nullptr;
        int closestDistance = std::numeric_limits<int>::max();
        for ( const Battle::Unit * enemy : enemies ) {
            if ( enemy && enemy->isValid() ) {
                int distance = arena.GetBoard()->GetDistance( currentUnit.GetPosition(), enemy->GetPosition() );
                if ( distance < closestDistance ) {
                    closestDistance = distance;
                    closestEnemy = enemy;
                }
            }
        }
        if ( closestEnemy ) {
            int attackTargetPosition = closestEnemy->GetHeadIndex();
            int positionNum = getClosestNeighborIndex( currentUnit, attackTargetPosition, arena );
            int attackDirection = Battle::Board::GetDirection( positionNum, attackTargetPosition );
            if ( currentUnit.GetShots() > 0 ) {
                attackDirection = -1; // swap to archery if available
                positionNum = -1;
            }
            // Validate attack parameters
            if ( CheckAttackParameters( &currentUnit, closestEnemy, positionNum, attackTargetPosition, attackDirection ) ) {
                actions.emplace_back( Battle::Command::ATTACK, uid, static_cast<int>( closestEnemy->GetUID() ), positionNum, attackTargetPosition, attackDirection );
            }
            else {
                actions.emplace_back( Battle::Command::SKIP, uid );
            }
            return actions;
        }
        actions.emplace_back( Battle::Command::SKIP, uid );
        return actions;
    }

    Battle::Actions DefendClosestAlly( Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        Battle::Actions actions;
        int uid = static_cast<int>( currentUnit.GetUID() );
        const Battle::Units allies( arena.GetCurrentForce().getUnits(), Battle::Units::REMOVE_INVALID_UNITS_AND_SPECIFIED_UNIT, &currentUnit );
        const Battle::Unit * closestAlly = nullptr;
        int closestDistance = std::numeric_limits<int>::max();
        for ( const Battle::Unit * ally : allies ) {
            if ( ally && ally->isValid() ) {
                int distance = arena.GetBoard()->GetDistance( currentUnit.GetPosition(), ally->GetPosition() );
                if ( distance < closestDistance ) {
                    closestDistance = distance;
                    closestAlly = ally;
                }
            }
        }
        if ( closestAlly ) {
            int defendTargetPosition = closestAlly->GetHeadIndex();
            int positionNum = getClosestNeighborIndex( currentUnit, defendTargetPosition, arena );
            int defendDirection = Battle::Board::GetDirection( positionNum, defendTargetPosition );
            // Validate defense parameters (if applicable)
            if ( CheckMoveParameters( &currentUnit, positionNum ) ) {
                actions.emplace_back( Battle::Command::MOVE, uid, positionNum );
            }
            else {
                actions.emplace_back( Battle::Command::SKIP, uid );
            }
            return actions;
        }
        actions.emplace_back( Battle::Command::SKIP, uid );
        return actions;
    }

    std::vector<int> selectTopActions( std::shared_ptr<QNetwork> model, const torch::Tensor & state )
    {
        auto q_values = model->get()->forward( state.unsqueeze( 0 ) ).squeeze( 0 ); // [ACTION_SIZE]
        auto accessor = q_values.accessor<float, 1>();

        std::vector<int> idx( ACTION_SIZE );
        std::iota( idx.begin(), idx.end(), 0 );

        std::sort( idx.begin(), idx.end(), [&]( int a, int b ) {
            return accessor[a] > accessor[b]; // largest Q first
        } );

        return idx; // sorted list of best to worst
    }

    int selectActionEpsilonGreedy( std::shared_ptr<QNetwork> model, torch::Tensor state_tensor )
    {
        std::uniform_real_distribution<double> dist( 0.0, 1.0 );
        static std::mt19937 rng( std::random_device{}() );
        double sample = dist( rng );

        // Random case – return ALL actions shuffled
        if ( sample < epsilon || !model ) {
            std::vector<int> a( ACTION_SIZE );
            std::iota( a.begin(), a.end(), 0 );
            std::shuffle( a.begin(), a.end(), rng );
            return a[0];
        }

        // Greedy case – sorted best→worst
        auto sorted = selectTopActions( model, state_tensor );
        return sorted[0];
    }

    // --- Memory API ---
    void remember_experience( const torch::Tensor & state, const torch::Tensor & prev_state, int64_t action, double reward, bool done, int color, bool isRanged )
    {
        Experience e;
        e.state = state.detach().to( torch::kCPU );
        e.prev_state = prev_state.detach().to( torch::kCPU );
        e.action = action;
        e.reward = reward;
        e.done = done;

        if ( color == 0x01 ) {
            g_replay_buffer_blue->push( e );
            // if ( reward > 0 ) {
            //     g_replay_buffer_red->set_last_reward( 1000.0f - reward );
            // }
        }
        else if ( color == 0x04 ) {
            g_replay_buffer_red->push( e );
            // if ( reward > 0 ) {
            //     //g_replay_buffer_blue->set_last_reward( 1000.0f - reward );
            // }
        }
    }

    // --- Optimization step ---
    void optimize_model( QNetwork & model, torch::optim::Optimizer & optimizer, std::shared_ptr<ReplayBuffer> replay_buffer, double gamma, torch::Device device )
    {
        if ( !replay_buffer )
            return;

        if ( replay_buffer->size() == 0 )
            return;

        // Take the entire buffer
        std::vector<NNAI::Experience> batch = replay_buffer->get_all();
        if ( batch.empty() )
            return;

        std::vector<torch::Tensor> states, next_states;
        std::vector<int64_t> actions;
        std::vector<float> rewards;
        std::vector<uint8_t> dones;

        for ( const auto & e : batch ) {
            states.push_back( e.state.to( device ) );
            next_states.push_back( e.prev_state.to( device ) );
            actions.push_back( e.action );
            rewards.push_back( static_cast<float>( e.reward ) );
            dones.push_back( e.done ? 1u : 0u );
            // std::cout << "[DBG-102] " << e.reward << std::endl;
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

        replay_buffer->clear();
    }

    void soft_update_target( QNetwork & local_model, QNetwork & target_model, double tau )
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

    // --- Battle-specific helpers retained from original file (reward, print, etc.) ---

    bool isNNControlled( int color )
    {
        if ( isRunningExperiments ) {
            switch ( color ) {
            case 0x01: // BLUE
                return true;
            case 0x04: // RED
                return false;
            }
        }
        return true;
    }

    int getClosestNeighborIndex( const Battle::Unit & unit, const int32_t targetIndex, Battle::Arena & arena )
    {
        int closestIndex = -1;
        uint32_t closestDistance = 9999;
        for ( Battle::CellDirection dir = Battle::TOP_LEFT; dir < Battle::CENTER; ++dir ) {
            int32_t neighborIndex = Battle::Board::GetIndexDirection( targetIndex, dir );
            uint32_t distance = arena.GetBoard()->GetDistance( unit.GetPosition(), neighborIndex );
            if ( distance < closestDistance ) {
                closestDistance = distance;
                closestIndex = neighborIndex;
            }
        }
        return closestIndex;
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

    float calculateReward( const torch::Tensor & initial_state, const torch::Tensor & curr_state, int color )
    {
        // --- CONSTANTS MATCHING FEATURE EXTRACTION ---
        constexpr int total_slots = 10; // 1 + 4 + 5
        constexpr int ally_slot_start = 0;
        constexpr int enemy_slot_start = 1 + 4;
        constexpr int hitpoint_idx = 0; // RAW HP index
        constexpr float EPS = 1e-6f;

        if ( !curr_state.defined() || curr_state.numel() == 0 ) {
            std::cout << "[DBG-002] curr_state undefined or empty -> return 0" << std::endl;
            return 0.0f;
        }

        torch::Tensor initial = initial_state.detach().cpu().contiguous();
        torch::Tensor cur = curr_state.detach().cpu().contiguous();

        int64_t elems = initial.numel();
        int feature_size = static_cast<int>( elems / total_slots );

        auto sumHp = [&]( const torch::Tensor & t, int start ) {
            float sum = 0.0f;
            int64_t total = t.numel();

            int count = 5;

            for ( int s = 0; s < count; ++s ) {
                int idx = ( start + s ) * feature_size + hitpoint_idx;

                if ( idx < 0 || idx >= total ) {
                    std::cout << "[DBG-011]  slot=" << s << " idx=" << idx << " OUT OF RANGE" << std::endl;
                    continue;
                }

                float v = t[idx].item<float>();

                if ( !std::isfinite( v ) ) {
                    std::cout << "[DBG-012]  slot=" << s << " idx=" << idx << " NON-FINITE value -> forced 0" << std::endl;
                    v = 0.0f;
                }

                sum += v;
            }
            return sum;
        };

        // --- sums ---
        float init_enemy_hp = sumHp( initial, enemy_slot_start );
        float curr_enemy_hp = sumHp( cur, enemy_slot_start );

        float init_allies_hp = sumHp( initial, ally_slot_start );

        float curr_allies_hp = sumHp( cur, ally_slot_start );

        // --- % of allies left (0..1) ---
        float ally_pct = ( init_allies_hp > EPS ) ? ( curr_allies_hp / init_allies_hp ) : 1.0f;

        // --- % of enemies left (0..1) ---
        float enemy_pct = ( init_enemy_hp > EPS ) ? ( curr_enemy_hp / init_enemy_hp ) : 1.0f;

        ally_pct = std::clamp( ally_pct, 0.0f, 1.0f );
        enemy_pct = std::clamp( enemy_pct, 0.0f, 1.0f );

        // --- final reward ---
        float reward = 1000.0f * ( 1.0f - enemy_pct ) * ally_pct;

        if ( !NNAI::skipDebugLog ) {
            std::cout << "[DBG-051] SUMMARY | prev_en=" << init_enemy_hp << " curr_en=" << curr_enemy_hp << " ally_pct=" << ally_pct << " reward=" << reward << std::endl;
        }

        // std::cout << "[DBG-999] RETURN reward=" << reward << std::endl;
        return reward;
    }

} // namespace Battle
