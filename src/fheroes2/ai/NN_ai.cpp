#include "NN_ai.h"

#pragma warning( disable : 4996 )

#include <filesystem>
#include <iostream>

#include <torch/torch.h>

#include "battle.h"
#include "battle_command.h"
// #include "battle_action.h"
#include <algorithm> // For std::reverse
#include <random>
#include <tuple>
#include <vector>

#include "battle_arena.h"
#include "battle_army.h"
#include "game.h"
#include "ui_tool.h"

namespace NNAI
{
    std::shared_ptr<BattleCNN> g_model1 = nullptr;
    std::shared_ptr<BattleCNN> g_model2 = nullptr;
    // Global model pointers for each color
    std::shared_ptr<BattleCNN> g_model_blue = nullptr;
    std::shared_ptr<BattleCNN> g_model_green = nullptr;
    std::shared_ptr<BattleCNN> g_model_red = nullptr;

    std::vector<torch::Tensor> g_states1;
    std::vector<std::vector<torch::Tensor>> g_actions1( HeadCount );
    std::vector<torch::Tensor> g_rewards1;
    std::vector<torch::Tensor> g_states2;
    std::vector<std::vector<torch::Tensor>> g_actions2( HeadCount );
    std::vector<torch::Tensor> g_rewards2;

    bool isTraining = true; // Defines if post battle dialog will open or the training loop will continue
    bool skipDebugLog = true; // Defines if post battle dialog will open or the training loop will continue
    bool isComparing = true; // Defines if game is comparing NNAI with Original AI

    int m1WinCount = 0;
    int m2WinCount = 0;

    torch::Device device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU );

    // Previous state tracking for reward calculation
    int prevEnemyHP1 = -1, prevAllyHP1 = -1, prevEnemyUnits1 = -1, prevAllyUnits1 = -1;
    int prevEnemyHP2 = -1, prevAllyHP2 = -1, prevEnemyUnits2 = -1, prevAllyUnits2 = -1;

    void initializeGlobalModels()
    {
        loadModel( NNAI::g_model_blue, "model_blue.pt" );
        loadModel( NNAI::g_model_green, "model_green.pt" );
        loadModel( NNAI::g_model_red, "model_red.pt" );
    }

    void createAndSaveModel( const std::string & model_path )
    {
        int64_t input_size = INPUT_SIZE, hidden_size = HIDDEN_SIZE, num_layers = LAYER_NUM;

        try {
            BattleCNN model;
            // BattleLSTM model( input_size, hidden_size, num_layers );
            torch::save( model, model_path );
        }
        catch ( const std::exception & e ) {
            std::cerr << "Error creating or saving the model: " << e.what() << std::endl;
        }
    }

    void saveModel( const BattleCNN & model, const std::string & model_path )
    {
        try {
            torch::save( model, model_path );
            std::cout << "Model saved to " << model_path << std::endl;
        }
        catch ( const std::exception & e ) {
            std::cerr << "Error saving the model: " << e.what() << std::endl;
        }
    }

    void loadModel( std::shared_ptr<BattleCNN> & modelPtr, const std::string & model_path )
    {
        namespace fs = std::filesystem;
        try {
            if ( !fs::exists( model_path ) ) {
                std::cerr << "Model file does not exist at " << model_path << ". Creating new model..." << std::endl;
                createAndSaveModel( model_path );
            }
            modelPtr = std::make_shared<BattleCNN>();
            torch::load( *modelPtr, model_path );
            modelPtr->get()->to( device ); // Move model to device after loading
            std::cout << "Model loaded from " << model_path << std::endl;
        }
        catch ( const std::exception & e ) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            modelPtr = nullptr;
        }
    }

    std::shared_ptr<BattleCNN> getModelByColor( int color )
    {
        switch ( color ) {
        case 0x01: // BLUE
            return g_model1;
        case 0x04: // RED
            return g_model2;
        default:
            std::cerr << "Warning: Unrecognized color " << color << ". Returning default model." << std::endl;
            return nullptr;
        }
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
        return true; // TODO Placeholder for actual logic to determine if the AI is controlled by NN
    }

    Battle::Actions planUnitTurn( Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        if ( currentUnit.Modes( Battle::TR_MOVED ) || currentUnit.GetCount() == 0 ) {
            return {};
        }

        BattleCNN & model = *getModelByColor( currentUnit.GetColor() );
        if ( !model ) {
            std::cerr << "Error: Neural network model is not initialized!" << std::endl;
            return {};
        }

        // Prepare input: [1, 25, 11, 9]
        torch::Tensor input = prepareBattleCNNInput( arena, currentUnit ).to( NNAI::device );

        // Forward pass
        std::vector<torch::Tensor> nn_output = model->forward( input );

        // Sample actions from each head
        std::vector<int64_t> nn_actions;
        for ( auto & head_output : nn_output ) {
            auto probs = torch::nn::functional::softmax( head_output, 1 ).nan_to_num( 0.0, 0.0, 0.0 );

            if ( !probs.isfinite().all().item<bool>() || probs.min().item<float>() < 0 ) {
                std::cerr << "Invalid probabilities detected, skipping unit." << std::endl;
                return {};
            }

            auto sampled = probs.multinomial( 1 );
            nn_actions.push_back( sampled.item<int64_t>() );
        }

        // Store actions if training
        if ( NNAI::isTraining ) {
            std::vector<torch::Tensor> head_actions;
            for ( auto val : nn_actions ) {
                head_actions.push_back( torch::tensor( val, torch::TensorOptions().dtype( torch::kLong ).device( NNAI::device ) ) );
            }

            auto & g_actions = ( currentUnit.GetColor() == 0x01 ) ? NNAI::g_actions1 : NNAI::g_actions2;
            if ( head_actions.size() != HeadCount ) {
                std::cerr << "Warning: head count mismatch" << std::endl;
            }
            else {
                for ( size_t h = 0; h < HeadCount; ++h ) {
                    g_actions[h].push_back( head_actions[h].clone().detach() );
                }
            }
        }

        // Map neural outputs to game actions
        int actionType = static_cast<int>( nn_actions[0] );
        int moveX = static_cast<int>( nn_actions[1] );
        int moveY = static_cast<int>( nn_actions[2] );
        int attackX = static_cast<int>( nn_actions[3] );
        int attackY = static_cast<int>( nn_actions[4] );

        int movePos = getIndexFromXY( moveX, moveY );
        int attackPos = getIndexFromXY( attackX, attackY );
        int attackDir = Battle::Board::GetDirection( movePos, attackPos );

        int currentUID = static_cast<int>( currentUnit.GetUID() );

        // Adjust action type if necessary
        if ( actionType == 0 && movePos == currentUnit.GetPosition().GetHead()->GetIndex() ) {
            actionType = 3; // SKIP
        }
        if ( actionType == 2 ) {
            actionType = 1; // Treat SPELLCAST as ATTACK
        }

        // Determine target UID
        int targetUID = -1;
        const auto * targetCell = arena.GetBoard()->GetCell( attackPos );
        if ( targetCell && targetCell->GetUnit() ) {
            targetUID = targetCell->GetUnit()->GetUID();
        }

        // Handle archery attack
        if ( currentUnit.GetShots() > 0 ) {
            attackDir = -1;
            movePos = -1;
        }

        // Validate ATTACK
        if ( actionType == 1
             && ( targetUID == -1 || !CheckAttackParameters( &currentUnit, targetCell ? targetCell->GetUnit() : nullptr, movePos, attackPos, attackDir ) ) ) {
            actionType = 0; // fallback to MOVE
        }

        // Validate MOVE
        if ( actionType == 0 && !CheckMoveParameters( &currentUnit, movePos ) ) {
            actionType = 3; // fallback to SKIP
        }

        // Build final actions
        Battle::Actions actions;
        switch ( actionType ) {
        case 0: // MOVE
            actions.emplace_back( Battle::Command::MOVE, currentUID, movePos );
            break;
        case 1: // ATTACK
            actions.emplace_back( Battle::Command::ATTACK, currentUID, targetUID, movePos, attackPos, attackDir );
            break;
        case 3: // SKIP
        default:
            actions.emplace_back( Battle::Command::SKIP, currentUID );
            break;
        }

        return actions;
    }

    // Extract features for a single unit
    std::vector<float> extractUnitFeatures( const Battle::Unit & unit, const Battle::Arena & arena, const Battle::Unit & currentunit )
    {
        std::vector<float> features;
        // flag for emptiness (0.0 since this is a real unit)
        features.push_back( 0.0f );

        std::pair<int, int> coords = getXYCoordinates( unit );

        features.push_back( static_cast<float>( unit.GetUID() ) ); // Unique ID
        features.push_back( normalize( static_cast<float>( coords.first ), 0, 8 ) ); // Position X (normalized by battlefield size)
        features.push_back( normalize( static_cast<float>( coords.second ), 0, 10 ) ); // Position Y (normalized by battlefield size)
        features.push_back( normalize( static_cast<float>( unit.GetCount() ), 0, 300 ) ); // Normalize Count
        features.push_back( normalize( static_cast<float>( unit.GetHitPoints() ), 0, 500 ) ); // Normalize HP
        features.push_back( normalize( static_cast<float>( unit.GetSpeed( false, true ) ), 0, 10 ) ); // Normalize speed
        features.push_back( normalize( static_cast<float>( arena.GetBoard()->GetDistance( currentunit.GetPosition(), unit.GetPosition() ) ), 0,
                                       50 ) ); // Distance to current unit
        features.push_back( normalize( static_cast<float>( unit.GetAttack() ), 0, 100 ) ); // Normalize attack
        features.push_back( normalize( static_cast<float>( unit.GetDefense() ), 0, 100 ) ); // Normalize defense
        features.push_back( unit.isFlying() ? 1.0f : 0.0f ); // Is flying
        features.push_back( unit.isArchers() ? 1.0f : 0.0f ); // Is archer
        features.push_back( normalize( static_cast<float>( unit.GetShots() ), 0, 50 ) ); // Normalize shots left
        features.push_back( unit.isHandFighting() ? 1.0f : 0.0f ); // Is hand fighting
        features.push_back( unit.isWide() ? 1.0f : 0.0f ); // Is wide
        features.push_back( unit.isAffectedByMorale() ? 1.0f : 0.0f ); // Affected by morale
        features.push_back( unit.isImmovable() ? 1.0f : 0.0f ); // Is immovable
        features.push_back( normalize( static_cast<float>( unit.GetMorale() ), 0, 100 ) ); // Morale
        features.push_back( normalize( static_cast<float>( unit.GetLuck() ), 0, 100 ) ); // Luck
        features.push_back( static_cast<float>( unit.GetColor() ) ); // Ally or enemy color

        currentunit.GetColor() == unit.GetColor() ? features.push_back( 1.0f ) : features.push_back( 0.0f ); // Is current unit ally or foe
        unit.GetColor() == arena.GetArmy1Color() ? features.push_back( 1.0f ) : features.push_back( 0.0f ); // Left or right
        unit.Modes( Battle::TR_MOVED ) ? features.push_back( 1.0f ) : features.push_back( 0.0f ); // Moved this turn
        unit.Modes( Battle::TR_RESPONDED ) ? features.push_back( 1.0f ) : features.push_back( 0.0f ); // Responded this turn
        arena.GetBoard()->CanAttackTargetFromPosition( currentunit, unit, arena.GetBoard()->GetDistance( currentunit.GetPosition(), unit.GetPosition() ) )
            ? features.push_back( 1.0f )
            : features.push_back( 0.0f ); // Can attack target from position

        return features;
    }

    torch::Tensor prepareBattleCNNInput( const Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        const int H = 11; // height (Y: 0..10)
        const int W = 9; // width  (X: 0..8)
        const int featureSize = 25; // now includes is_empty

        torch::Tensor input = torch::zeros( { featureSize, H, W }, torch::TensorOptions().dtype( torch::kFloat32 ).device( NNAI::device ) );

        for ( int y = 0; y < H; ++y ) {
            for ( int x = 0; x < W; ++x ) {
                const Battle::Cell * cell = arena.GetBoard()->GetCell( getIndexFromXY( x, y ) );

                if ( const Battle::Unit * unit = cell->GetUnit() ) {
                    if ( unit->isValid() ) {
                        std::vector<float> feats = extractUnitFeatures( *unit, arena, currentUnit );
                        for ( int c = 0; c < featureSize; ++c ) {
                            input[c][y][x] = feats[c];
                        }
                    }
                }
                else {
                    // Empty tile → mark only is_empty = 1.0f
                    input[0][y][x] = 1.0f;
                }
            }
        }

        input = input.unsqueeze( 0 ); // [1, 25, 11, 9]
        // [25, 11, 9]

        return input;
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
    std::tuple<BattleCNN &, std::string, BattleCNN &, std::string, BattleCNN &, std::string> SelectRandomModels()
    {
        // Pair each model pointer with its name
        std::vector<std::pair<std::shared_ptr<BattleCNN>, std::string>> models = { { g_model_blue, "blue" }, { g_model_green, "green" }, { g_model_red, "red" } };

        // Remove nullptrs
        models.erase( std::remove_if( models.begin(), models.end(), []( const auto & m ) { return !m.first; } ), models.end() );

        if ( models.size() < 2 ) {
            throw std::runtime_error( "Not enough models to select two random ones." );
        }

        std::random_device rd;
        std::mt19937 gen( rd() );
        std::uniform_int_distribution<> dis( 0, static_cast<int>( models.size() ) - 1 );

        int idx1 = dis( gen );
        int idx2;
        do {
            idx2 = dis( gen );
        } while ( idx2 == idx1 );

        int idx3;
        do {
            idx3 = dis( gen );
        } while ( idx3 == idx1 || idx3 == idx2 );

        return std::tie( *models[idx1].first, models[idx1].second, *models[idx2].first, models[idx2].second, *models[idx3].first, models[idx3].second );
    }

    void tryTrainModel( BattleCNN & model, torch::optim::Optimizer & optimizer, const std::vector<torch::Tensor> & states,
                        const std::vector<std::vector<torch::Tensor>> & actions, const std::vector<torch::Tensor> & rewards, float & total_loss,
                        float & epoch_total_reward, torch::Device device, int model_id )
    {
        if ( states.empty() || rewards.empty() || actions.empty() ) {
            std::cout << "states: " << states.size() << " rewards: " << rewards.size() << " actions: " << actions.size()
                      << " actions[0]: " << ( actions.empty() ? 0 : actions[0].size() ) << "\n";
            std::cout << "Empty states, rewards, or actions. Skipping training for model " << model_id << "\n";
            return;
        }

        // --- Flatten batch dimension if needed ---
        std::vector<torch::Tensor> flat_states;
        flat_states.reserve( states.size() );
        for ( auto & s : states ) {
            if ( s.dim() == 4 && s.size( 0 ) == 1 ) {
                // Remove singleton batch dim → [C, H, W]
                flat_states.push_back( s.squeeze( 0 ) );
            }
            else {
                flat_states.push_back( s );
            }
        }

        // Stack into proper 4D tensor: [batch, channels, H, W]
        torch::Tensor state_batch = torch::stack( flat_states ).to( device ); // shape [N, 25, 11, 9]

        // Stack rewards
        torch::Tensor reward_batch = torch::stack( rewards ).to( device ).to( torch::kFloat ).view( { -1 } );

        // Stack action tensors
        std::vector<torch::Tensor> action_batches;
        action_batches.reserve( actions.size() );
        for ( size_t h = 0; h < actions.size(); ++h ) {
            TORCH_CHECK( !actions[h].empty(), "actions[", h, "] is empty" );
            auto ab = torch::stack( actions[h] ).to( device, torch::kLong );
            action_batches.push_back( ab );
        }

        optimizer.zero_grad();

        // --- Forward pass ---
        auto logits = model->forward( state_batch );

        // --- Compute discounted returns ---
        const float gamma = 0.99f;
        std::vector<float> discounted( reward_batch.size( 0 ) );
        float running_return = 0.0f;
        for ( int64_t t = reward_batch.size( 0 ) - 1; t >= 0; --t ) {
            running_return = reward_batch[t].item<float>() + gamma * running_return;
            discounted[t] = running_return;
        }
        auto returns = torch::tensor( discounted, reward_batch.options() );

        // Scale and normalize
        const float max_reward = 1100.0f;
        returns = returns / max_reward;
        auto mean = returns.mean().detach();
        auto std = returns.std( false ).detach();
        auto norm_rewards = ( returns - mean ) / ( std + 1e-6f );

        // --- Loss computation ---
        torch::Tensor loss = torch::zeros( {}, torch::TensorOptions().dtype( torch::kFloat32 ).device( device ) );
        const float entropy_coef = 0.05f;

        for ( size_t h = 0; h < logits.size(); ++h ) {
            auto log_prob = torch::nn::functional::log_softmax( logits[h], 1 );
            auto idx = action_batches[h].unsqueeze( 1 ); // [B,1]
            auto selected_log_prob = log_prob.gather( 1, idx ).squeeze( 1 ); // [B]

            // Entropy regularization
            auto prob = torch::exp( log_prob );
            auto entropy = -( prob * log_prob ).sum( 1 ).mean();

            auto policy_loss = -( selected_log_prob * norm_rewards ).mean();
            loss += policy_loss - entropy_coef * entropy;
        }

        // --- Backprop ---
        loss.backward();
        optimizer.step();

        total_loss += loss.detach().cpu().item<double>();

        // Track total reward
        float reward_sum = 0.0f;
        for ( const auto & r : rewards )
            reward_sum += r.cpu().item<float>();
        epoch_total_reward += reward_sum;
    }

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

} // NNAI

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

    float calculateReward( const Battle::Arena & currArena, int color )
    {
        float reward = 0.0f;

        if ( !NNAI::skipDebugLog )
            std::cout << "\n[DEBUG] calculateReward: color=" << color << std::endl;

        // Select previous values
        int * prevEnemyHP;
        int * prevEnemyUnits;
        if ( color == currArena.GetArmy1Color() ) {
            prevEnemyHP = &NNAI::prevEnemyHP1;
            prevEnemyUnits = &NNAI::prevEnemyUnits1;
        }
        else {
            prevEnemyHP = &NNAI::prevEnemyHP2;
            prevEnemyUnits = &NNAI::prevEnemyUnits2;
        }

        // Get current values
        int currEnemyHP = currArena.getEnemyForce( color ).GetAliveHitPoints();
        int totalEnemyHP = currArena.getEnemyForce( color ).GetTotalHitPoints();
        int currEnemyUnits = currArena.getEnemyForce( color ).GetAliveCounts();
        int currAllyHP = currArena.getForce( color ).GetAliveHitPoints();
        int currAllyUnits = currArena.getForce( color ).GetAliveCounts();

        if ( !NNAI::skipDebugLog ) {
            std::cout << "[DEBUG] Current: EnemyHP=" << currEnemyHP << ", AllyHP=" << currAllyHP << ", EnemyUnits=" << currEnemyUnits << ", AllyUnits=" << currAllyUnits
                      << std::endl;

            std::cout << "[DEBUG] Previous: EnemyHP=" << *prevEnemyHP << ", EnemyUnits=" << *prevEnemyUnits << std::endl;
        }

        // Only calculate reward if not first turn
        if ( *prevEnemyHP != -1 ) {
            float deltaEnemyHP = static_cast<float>( *prevEnemyHP - currEnemyHP );
            reward += 100.0f * deltaEnemyHP / static_cast<float>( totalEnemyHP ); // Damage dealt in percent

            if ( !NNAI::skipDebugLog )
                std::cout << "[DEBUG] Delta: EnemyHP=" << deltaEnemyHP << ", PartialReward=" << reward << std::endl;
        }

        reward = std::max( reward, 0.0f );

        // Win condition
        if ( currEnemyHP == 0 ) {
            reward += 100;
            if ( !NNAI::skipDebugLog )
                std::cout << "[DEBUG] Win detected: Enemy defeated." << std::endl;
            if ( color == currArena.GetArmy1Color() ) {
                NNAI::m1WinCount++;
            }
            else {
                NNAI::m2WinCount++;
            }
        }

        // Update for next turn
        *prevEnemyHP = currEnemyHP;
        *prevEnemyUnits = currEnemyUnits;
        if ( color == currArena.GetArmy1Color() ) {
            NNAI::prevAllyHP1 = currAllyHP;
            NNAI::prevAllyUnits1 = currAllyUnits;
        }
        else {
            NNAI::prevAllyHP2 = currAllyHP;
            NNAI::prevAllyUnits2 = currAllyUnits;
        }

        if ( !NNAI::skipDebugLog )
            std::cout << "[DEBUG] Final reward for color " << color << ": " << reward << std::endl;

        return reward;
    }
}
