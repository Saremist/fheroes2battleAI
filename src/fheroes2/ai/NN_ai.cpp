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

#include "battle_arena.h"
#include "battle_army.h"
#include "game.h"
#include "ui_tool.h"

namespace NNAI
{
    std::shared_ptr<BattleLSTM> g_model1 = nullptr;
    std::shared_ptr<BattleLSTM> g_model2 = nullptr;
    // Global model pointers for each color
    std::shared_ptr<BattleLSTM> g_model_blue = nullptr;
    std::shared_ptr<BattleLSTM> g_model_green = nullptr;
    std::shared_ptr<BattleLSTM> g_model_red = nullptr;
    std::shared_ptr<BattleLSTM> g_model_yellow = nullptr;
    std::shared_ptr<BattleLSTM> g_model_orange = nullptr;
    std::shared_ptr<BattleLSTM> g_model_purple = nullptr;

    std::vector<torch::Tensor> * g_states1 = nullptr;
    std::vector<std::vector<torch::Tensor>> * g_actions1 = nullptr;
    std::vector<torch::Tensor> * g_rewards1 = nullptr;
    std::vector<torch::Tensor> * g_states2 = nullptr;
    std::vector<std::vector<torch::Tensor>> * g_actions2 = nullptr;
    std::vector<torch::Tensor> * g_rewards2 = nullptr;

    const int TrainingLoopsCount = 1;

    int m1skipCount = 0;
    int m1CorrectMovesCount = 0;
    int m1turnCount = 0;
    int m2skipCount = 0;
    int m2CorrectMovesCount = 0;
    int m2turnCount = 0;

    torch::Device device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU );

    // Previous state tracking for reward calculation
    int prevEnemyHP1 = -1, prevAllyHP1 = -1, prevEnemyUnits1 = -1, prevAllyUnits1 = -1;
    int prevEnemyHP2 = -1, prevAllyHP2 = -1, prevEnemyUnits2 = -1, prevAllyUnits2 = -1;

    void initializeGlobalModels()
    {
        loadModel( NNAI::g_model_blue, "model_blue.pt" );
        loadModel( NNAI::g_model_green, "model_green.pt" );
        loadModel( NNAI::g_model_red, "model_red.pt" );
        // loadModel( NNAI::g_model_yellow, "model_yellow.pt" );
        // loadModel( NNAI::g_model_orange, "model_orange.pt" );
        // loadModel( NNAI::g_model_purple, "model_purple.pt" );
    }

    void createAndSaveModel( const std::string & model_path )
    {
        int64_t input_size = 15;
        int64_t hidden_size = 256;
        int64_t num_layers = 2;

        try {
            BattleLSTM model( input_size, hidden_size, num_layers );
            torch::save( model, model_path );
        }
        catch ( const std::exception & e ) {
            std::cerr << "Error creating or saving the model: " << e.what() << std::endl;
        }
    }

    void saveModel( const BattleLSTM & model, const std::string & model_path )
    {
        try {
            torch::save( model, model_path );
            std::cout << "Model saved to " << model_path << std::endl;
        }
        catch ( const std::exception & e ) {
            std::cerr << "Error saving the model: " << e.what() << std::endl;
        }
    }

    void loadModel( std::shared_ptr<BattleLSTM> & modelPtr, const std::string & model_path )
    {
        namespace fs = std::filesystem;
        try {
            if ( !fs::exists( model_path ) ) {
                std::cerr << "Model file does not exist at " << model_path << ". Creating new model..." << std::endl;
                createAndSaveModel( model_path );
            }
            modelPtr = std::make_shared<BattleLSTM>();
            torch::load( *modelPtr, model_path );
            modelPtr->get()->to( device ); // Move model to device after loading
            std::cout << "Model loaded from " << model_path << std::endl;
        }
        catch ( const std::exception & e ) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            modelPtr = nullptr;
        }
    }

    std::shared_ptr<BattleLSTM> getModelByColor( int color )
    {
        switch ( color ) {
        case 0x01: // BLUE
            return g_model1;
        case 0x04: // RED
            return g_model2;
        default:
            std::cerr << "Warning: Unrecognized color " << color << ". Returning default model." << std::endl;
        }
    }

    bool isNNControlled( int color )
    {
        return true; // TODO Placeholder for actual logic to determine if the AI is controlled by NN
    }

    // Converts a vector of action indices (from NN output) to Battle::Actions
    Battle::Actions planUnitTurn( Battle::Arena & arena, const Battle::Unit & currentUnit ) // , const Battle::Arena & arena
    {
        if ( currentUnit.Modes( Battle::TR_MOVED ) ) {
            return {};
        }

        BattleLSTM & model = *getModelByColor( currentUnit.GetColor() );

        if ( !model ) {
            std::cerr << "Error: Neural network model (g_model1) is not initialized!" << std::endl;
            return {};
        }

        torch::Tensor input = prepareBattleLSTMInput( arena, currentUnit );
        if ( currentUnit.GetCount() == 0 )
            return {};

        std::vector<torch::Tensor> nn_output = model->forward( input ); // Get the NN output

        // Extract highest-probability action per head
        std::vector<int64_t> nn_outputs;
        for ( const auto & head_output : nn_output ) {
            // Get probabilities
            auto probs = torch::nn::functional::softmax( head_output, /*dim=*/1 );
            probs = probs.clamp( 0, 1 );
            probs = probs.nan_to_num( 0.0, 0.0, 0.0 );
            if ( !probs.isfinite().all().item<bool>() || probs.min().item<float>() < 0 ) {
                std::cerr << "Invalid probabilities detected, SKIPPING." << std::endl;
                return {}; // Skip if probabilities are not valid
            }
            else {
                auto sampled = probs.multinomial( /*num_samples=*/1 );
                nn_outputs.push_back( sampled.item<int64_t>() );
            }
        }

        if ( NNAI::isTraining ) {
            const uint8_t color = currentUnit.GetColor();
            std::vector<torch::Tensor> head_actions;

            for ( int64_t val : nn_outputs ) {
                head_actions.push_back( torch::tensor( val, torch::TensorOptions().dtype( torch::kLong ).device( NNAI::device ) ) );
            }

            if ( head_actions.size() != 3 ) {
                std::cerr << "Warning: Expected 3 heads, got " << head_actions.size() << std::endl;
                return {};
            }

            // Remove batch dimension from input if needed
            torch::Tensor squeezed_input = input.squeeze( 0 );

            if ( color == 0x01 ) { // BLUE team
                g_states1->push_back( squeezed_input.to( NNAI::device ) );
                for ( size_t h = 0; h < 3; ++h ) {
                    ( *g_actions1 )[h].push_back( head_actions[h].to( NNAI::device ) );
                }
            }
            else if ( color == 0x04 ) { // RED team
                g_states2->push_back( squeezed_input.to( NNAI::device ) );
                for ( size_t h = 0; h < 3; ++h ) {
                    ( *g_actions2 )[h].push_back( head_actions[h].to( NNAI::device ) );
                }
            }
            else {
                std::cerr << "Warning: Unrecognized color " << static_cast<int>( color ) << ". Skipping unit." << std::endl;
            }
        }

        Battle::Actions actions;

        int actionType = static_cast<int>( nn_outputs[0] ); // Action type index (0: SKIP, 1: MOVE, 2: ATTACK, 3: SPELLCAST)
        int positionNum = static_cast<int>( nn_outputs[1] ); // Position index
        int attack_direction = static_cast<int>( nn_outputs[2] ); // Direction index (0-6)

        attack_direction = ( attack_direction >= 6 ? -1 : 1 << attack_direction ); // Convert to actual direction (1,2,4,8,16,32) or -1 for archery
        int currentUnitUID = currentUnit.GetUID(); // Current unit UID

        // int attackTargetPositon = apply_attack_to_grid(positionNum, attack_direction);
        int attackTargetPositon = arena.GetBoard()->GetIndexDirection( positionNum, attack_direction );

        int targetUnitUID = -1;
        auto cell = arena.GetBoard()->GetCell( attackTargetPositon );
        if ( cell ) {
            auto unit = cell->GetUnit();
            if ( unit ) {
                targetUnitUID = unit->GetUID();
            }
        }

        if ( positionNum == currentUnit.GetPosition().GetHead()->GetIndex() ) {
            positionNum = -1; // Attack in place required format
        }

        if ( positionNum == -1 && actionType == 0 ) {
            actionType = 3; // SKIP move to already ocupied slot
            // std::cout << "Already at " << currentUnit.GetPosition().GetHead() << ". Defaulting to SKIP." << std::endl;
        }

        /*std::cout << std::endl
                  << "Action Type: " << actionType << ", Position Num: " << positionNum << ", Attack Direction: " << attack_direction
                  << ", Current Unit UID: " << currentUnitUID << ", Target unit UID: " << targetUnitUID << ", Attack Target Position: " << attackTargetPositon
                  << std::endl;*/

        if ( actionType == 1
             && ( targetUnitUID == -1
                  || !CheckAttackParameters( &currentUnit, arena.GetBoard()->GetCell( attackTargetPositon )->GetUnit(), positionNum, attackTargetPositon,
                                             attack_direction ) ) ) {
            actionType = 0; // Move instead of attack if attack would be illegal
            // std::cout << "Attack to position " << attackTargetPositon << " is not available for unit " << currentUnitUID << ". Changing to MOVE." << std::endl;
        }
        if ( actionType == 0 && !CheckMoveParameters( &currentUnit, positionNum ) ) {
            actionType = 3; // SKIP move to illegal positon ends up as SKIP
            // std::cout << "Position " << positionNum << " is not available for unit " << currentUnitUID << ". Defaulting to SKIP." << std::endl;
        }

        switch ( actionType ) {
        case 0:
            // MOVE: [CommandType, unitUID, targetCell
            actions.emplace_back( Battle::Command::MOVE, currentUnitUID, positionNum );
            break;
        case 1:
            // ATTACK: [CommandType, unitUID, targetUnitUID, moveTargetIdx, attackTargetIdx, attackDirection]
            actions.emplace_back( Battle::Command::ATTACK, currentUnitUID, targetUnitUID, positionNum, attackTargetPositon, attack_direction );
            // TODO
            break;
        case 2:
            // SPELLCAST: [CommandType, spellID, targetCell]
            // actions.emplace_back( Battle::Command::SPELLCAST, nn_output[1], nn_output[2] ); //Spellcast is disabled due to complexity exiting NN scope
            // actions.emplace_back( Battle::Command::SKIP, currentUnitUID )
            actionType = 3; // Spellcast is disabled due to complexity exiting NN scope
            // break;
        case 3:
            // SKIP: [CommandType, unitUID]
            actions.emplace_back( Battle::Command::SKIP, currentUnitUID );
            break;
        default:
            // Unknown command, fallback to SKIP
            actions.emplace_back( Battle::Command::SKIP, currentUnitUID );
            break;
        }

        if ( currentUnit.GetColor() == 0x01 ) { // separeate skip counts for models to penalise idling
            if ( actionType == 3 ) {
                m1skipCount++; // Penalty normalised
            }
            else {
                m1CorrectMovesCount++;
            }
            m1turnCount++;
        }
        else {
            if ( actionType == 3 ) {
                m2skipCount++;
            }
            else {
                m2CorrectMovesCount++;
            }
            m2turnCount++;
        }

        return actions;
    }

    // Normalize a value to the range [0, 1]
    inline float normalize( float value, float min, float max )
    {
        // Cast to float to avoid integer division and loss of precision
        return ( static_cast<float>( value ) - static_cast<float>( min ) ) / ( static_cast<float>( max ) - static_cast<float>( min ) );
    }

    // Extract features for a single unit
    std::vector<float> extractUnitFeatures( const Battle::Unit & unit, const Battle::Arena & arena )
    {
        std::vector<float> features;

        features.push_back( unit.GetUID() ); // Unique ID
        features.push_back( normalize( unit.GetHitPoints(), 0, 500 ) ); // Normalize HP
        features.push_back( normalize( unit.GetSpeed( false, true ), 0, 20 ) ); // Normalize speed
        features.push_back( normalize( unit.GetAttack(), 0, 100 ) ); // Normalize attack
        features.push_back( normalize( unit.GetDefense(), 0, 100 ) ); // Normalize defense
        features.push_back( unit.isFlying() ? 1.0f : 0.0f ); // Is flying
        features.push_back( unit.isArchers() ? 1.0f : 0.0f ); // Is archer
        features.push_back( unit.isHandFighting() ? 1.0f : 0.0f ); // Is hand fighting
        features.push_back( unit.isWide() ? 1.0f : 0.0f ); // Is wide
        features.push_back( unit.isAffectedByMorale() ? 1.0f : 0.0f ); // Affected by morale
        features.push_back( unit.isImmovable() ? 1.0f : 0.0f ); // Is immovable
        features.push_back( unit.GetMorale() ); // Morale
        features.push_back( unit.GetLuck() ); // Luck
        features.push_back( unit.GetColor() ); // Ally or enemy color
        features.push_back( normalize( unit.GetPosition().GetHead()->GetIndex(), 0, 100 ) ); // Position (normalized by battlefield size)

        return features;
    }

    torch::Tensor prepareBattleLSTMInput( const Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        // Get all units for both sides
        int _myColor = currentUnit.GetCurrentColor();

        const Battle::Units enemies( arena.getEnemyForce( arena.GetCurrentColor() ).getUnits(), Battle::Units::REMOVE_INVALID_UNITS_AND_SPECIFIED_UNIT, &currentUnit );
        const Battle::Units allies( arena.GetCurrentForce().getUnits(), Battle::Units::REMOVE_INVALID_UNITS_AND_SPECIFIED_UNIT, &currentUnit );

        // Prepare feature vectors
        std::vector<std::vector<float>> featuresList;

        // 1. Current unit (always first)
        featuresList.push_back( extractUnitFeatures( currentUnit, arena ) );

        // 2. Up to 4 other allies (excluding current unit)
        int allyCount = 0;
        for ( const Battle::Unit * unit : allies ) {
            if ( unit && unit->isValid() && unit != &currentUnit ) {
                featuresList.push_back( extractUnitFeatures( *unit, arena ) );
                ++allyCount;
                if ( allyCount == 4 )
                    break;
            }
        }
        // Pad with zeros if less than 4 allies
        if ( !featuresList.empty() ) {
            const int featureSize = static_cast<int>( featuresList[0].size() );
            while ( allyCount < 4 ) {
                featuresList.push_back( std::vector<float>( featureSize, 0.0f ) );
                ++allyCount;
            }
        }

        // 3. Up to 5 enemies
        int enemyCount = 0;
        for ( const Battle::Unit * unit : enemies ) {
            if ( unit && unit->isValid() ) {
                featuresList.push_back( extractUnitFeatures( *unit, arena ) );
                ++enemyCount;
                if ( enemyCount == 5 )
                    break;
            }
        }
        // Pad with zeros if less than 5 enemies
        if ( !featuresList.empty() ) {
            const int featureSize = static_cast<int>( featuresList[0].size() );
            while ( enemyCount < 5 ) {
                featuresList.push_back( std::vector<float>( featureSize, 0.0f ) );
                ++enemyCount;
            }
        }

        // Convert to torch tensor: [1, seq_len, input_size]
        if ( featuresList.empty() )
            return torch::empty( { 1, 0, 0 }, torch::TensorOptions().dtype( torch::kFloat32 ).device( NNAI::device ) );

        const int seq_len = static_cast<int>( featuresList.size() );
        const int input_size = static_cast<int>( featuresList[0].size() );

        torch::Tensor input = torch::zeros( { 1, seq_len, input_size }, torch::TensorOptions().dtype( torch::kFloat32 ).device( NNAI::device ) );
        for ( int i = 0; i < seq_len; ++i ) {
            for ( int j = 0; j < input_size; ++j ) {
                input[0][i][j] = featuresList[i][j];
            }
        }

        return input;
    }

    void trainingGameLoop( bool isFirstGameRun, bool isProbablyDemoVersion )
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
    std::tuple<BattleLSTM &, std::string, BattleLSTM &, std::string> SelectRandomModels()
    {
        // Pair each model pointer with its name
        std::vector<std::pair<std::shared_ptr<BattleLSTM>, std::string>> models
            = { { g_model_blue, "blue" },     { g_model_green, "green" },   { g_model_red, "red" }/*,
                { g_model_yellow, "yellow" }, { g_model_orange, "orange" }, { g_model_purple, "purple" }*/ };

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

        return std::tie( *models[idx1].first, models[idx1].second, *models[idx2].first, models[idx2].second );
    }

}

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

        if ( !NNAI::isTraining )
            std::cout << std::endl << "[DEBUG] calculateReward: color=" << color << std::endl;

        // Select which previous values to use
        int *prevEnemyHP, *prevAllyHP, *prevEnemyUnits, *prevAllyUnits;
        if ( color == currArena.GetArmy1Color() ) {
            prevEnemyHP = &NNAI::prevEnemyHP1;
            prevAllyHP = &NNAI::prevAllyHP1;
            prevEnemyUnits = &NNAI::prevEnemyUnits1;
            prevAllyUnits = &NNAI::prevAllyUnits1;
        }
        else {
            prevEnemyHP = &NNAI::prevEnemyHP2;
            prevAllyHP = &NNAI::prevAllyHP2;
            prevEnemyUnits = &NNAI::prevEnemyUnits2;
            prevAllyUnits = &NNAI::prevAllyUnits2;
        }

        // Get current values
        int currEnemyHP = currArena.getEnemyForce( color ).GetAliveHitPoints();
        int totalEnemyHP = currArena.getEnemyForce( color ).GetTotalHitPoints();
        int currAllyHP = currArena.getForce( color ).GetAliveHitPoints();
        int totalAllyHP = currArena.getForce( color ).GetTotalHitPoints();
        int currEnemyUnits = currArena.getEnemyForce( color ).GetAliveCounts();
        int currAllyUnits = currArena.getForce( color ).GetAliveCounts();

        if ( !NNAI::isTraining )
            std::cout << "[DEBUG] Current: EnemyHP=" << currEnemyHP << "/" << totalEnemyHP << ", AllyHP=" << currAllyHP << "/" << totalAllyHP
                      << ", EnemyUnits=" << currEnemyUnits << ", AllyUnits=" << currAllyUnits << std::endl;

        if ( !NNAI::isTraining )
            std::cout << "[DEBUG] Previous: EnemyHP=" << *prevEnemyHP << ", AllyHP=" << *prevAllyHP << ", EnemyUnits=" << *prevEnemyUnits
                      << ", AllyUnits=" << *prevAllyUnits << std::endl;

        // Only calculate reward if previous values are valid (not first turn)
        if ( *prevEnemyHP != -1 ) {
            float deltaEnemyHP = static_cast<float>( *prevEnemyHP ) - static_cast<float>( currEnemyHP );
            float deltaEnemyUnits = static_cast<float>( *prevEnemyUnits ) - static_cast<float>( currEnemyUnits );
            reward += 100.0f * deltaEnemyHP / static_cast<float>( totalEnemyHP ); // Damage dealt in percent
            reward += 25.0f * deltaEnemyUnits;

            if ( !NNAI::isTraining )
                std::cout << "[DEBUG] Delta: EnemyHP=" << deltaEnemyHP << ", EnemyUnits=" << deltaEnemyUnits << ", PartialReward=" << reward << std::endl;
        }

        // (Optional) Uncommented reward shaping for correct moves/skips can be logged here if re-enabled

        // Win condition
        if ( currEnemyHP <= 10 ) {
            if ( !NNAI::isTraining )
                std::cout << "[DEBUG] Win detected: currEnemyHP=" << currEnemyHP << " / " << totalEnemyHP << std::endl;
            reward += 500.0f;
        }

        // Update previous values for next turn
        *prevEnemyHP = currEnemyHP;
        *prevAllyHP = currAllyHP;
        *prevEnemyUnits = currEnemyUnits;
        *prevAllyUnits = currAllyUnits;

        if ( !NNAI::isTraining )
            std::cout << "[DEBUG] Final reward for color " << color << ": " << reward << std::endl;

        return reward;
    }
}
