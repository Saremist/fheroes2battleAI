#include "NN_ai.h"

#include <iostream>
#include <torch/torch.h>
#include <filesystem>
#include "battle.h"
#include "battle_command.h"
#include "battle_arena.h"
#include "battle_army.h"
#include <algorithm> // For std::reverse

#include <tuple>

#include "game.h"
#include "ui_tool.h"

namespace NNAI
{
    std::shared_ptr<NNAI::BattleLSTM> g_model1 = nullptr;
    std::shared_ptr<NNAI::BattleLSTM> g_model2 = nullptr;
    const bool isTraining = true; // Defines if post battle dialog will open
    const int TrainingLoopsCount = 1;

    torch::Device device( torch::kCPU );

    void initializeGlobalModels( int64_t input_size, int64_t hidden_size, int64_t num_layers )
    {
        g_model1 = std::make_shared<BattleLSTM>( input_size, hidden_size, num_layers );
        g_model2 = std::make_shared<BattleLSTM>( input_size, hidden_size, num_layers );
    }

   void createAndSaveModel( const std::string & model_path )
   {
       int64_t input_size = 15; 
       int64_t hidden_size = 128;
       int64_t num_layers = 1;

       try {
           BattleLSTM model( input_size, hidden_size, num_layers );
           torch::save( model, model_path );
       }
       catch ( const std::exception & e ) {
           std::cerr << "Error creating or saving the model: " << e.what() << std::endl;
       }
   }

    void trainModel( BattleLSTM & model, int64_t num_epochs, double learning_rate, torch::Device device, int64_t NUM_SELF_PLAY_GAMES )
   {
       model->train();
       torch::optim::Adam optimizer( model->parameters(), torch::optim::AdamOptions( learning_rate ) );

       for ( int64_t epoch = 0; epoch < num_epochs; ++epoch ) {
           double total_loss = 0.0;
           int game_count = 0;

           for ( int i = 0; i < NUM_SELF_PLAY_GAMES; ++i ) {
               std::vector<torch::Tensor> states;
               std::vector<std::vector<torch::Tensor>> actions_per_head( 5 ); // 5 heads
               std::vector<torch::Tensor> rewards;

               // simulateSelfPlayGame(model, states, actions_per_head, rewards, device); //TODO

               if ( states.empty() || rewards.empty() || actions_per_head[0].empty() ) {
                   std::cerr << "Warning: Empty game data. Skipping game " << i << std::endl;
                   continue;
               }

               // Stack state and reward batches
               torch::Tensor state_batch = torch::stack( states ).to( device ); // [batch, seq_len, input_size]
               torch::Tensor reward_batch = torch::stack( rewards ).to( device ); // [batch]

               // Stack action batches per head
               std::vector<torch::Tensor> action_batches;
               for ( int h = 0; h < 5; ++h ) {
                   action_batches.push_back( torch::stack( actions_per_head[h] ).to( device ) ); // [batch]
               }

               optimizer.zero_grad();

               // Forward pass through model
               std::vector<torch::Tensor> logits = model->forward( state_batch ); // 5 logits tensors

               if ( logits.size() != 5 || action_batches.size() != 5 ) {
                   std::cerr << "Error: Mismatch in number of heads." << std::endl;
                   continue;
               }

               // Compute loss across all heads
               torch::Tensor total_head_loss = torch::zeros( {}, torch::TensorOptions().dtype( torch::kFloat32 ).device( device ) );

               for ( size_t h = 0; h < 5; ++h ) {
                   torch::Tensor log_prob = torch::nn::functional::log_softmax( logits[h], /*dim=*/1 );
                   torch::Tensor selected_log_prob = log_prob.gather( 1, action_batches[h].unsqueeze( 1 ) ).squeeze( 1 );
                   total_head_loss -= ( selected_log_prob * reward_batch ).mean();
               }

               total_head_loss.backward();
               optimizer.step();

               total_loss += total_head_loss.item<double>();
               ++game_count;
           }

           if ( game_count > 0 )
               std::cout << "Epoch [" << ( epoch + 1 ) << "/" << num_epochs << "], Avg Loss: " << ( total_loss / game_count ) << std::endl;
           else
               std::cout << "Epoch [" << ( epoch + 1 ) << "/" << num_epochs << "], No games played." << std::endl;
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

    void loadModel( std::shared_ptr<BattleLSTM>& modelPtr, const std::string & model_path )
       {
           namespace fs = std::filesystem;
           try {
               if ( !fs::exists( model_path ) ) {
                   std::cerr << "Model file does not exist at " << model_path << ". Creating new model..." << std::endl;
                   createAndSaveModel( model_path );
               }
               modelPtr = std::make_shared<BattleLSTM>();
               torch::load( *modelPtr, model_path );
               std::cout << "Model loaded from " << model_path << std::endl;
           }
           catch ( const std::exception & e ) {
               std::cerr << "Error loading the model: " << e.what() << std::endl;
               modelPtr = nullptr;
           }
       }


    bool isNNControlled( int color )
    {
        return true; // Placeholder for actual logic to determine if the AI is controlled by NN
    }
    
    Battle::Actions planUnitTurn( Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        Battle::Actions actions = predict_action( currentUnit, arena );
        return actions;
    }


    std::tuple<int, int> grid_id_to_coordinates(int GridID)
    {
        if ( GridID < 1 || GridID > 99 ) {
            std::cerr << "GridID out of bounds!" << std::endl;
            return std::make_tuple( -1, -1 ); // Invalid coordinates
        }
        int x = (GridID-1) % 11;
        int y = (GridID-1) / 11;
        return std::make_tuple( x, y );
    }

    std::tuple<int, int> apply_attack_to_coordinates(std::tuple<int, int> GridCoords, int AttackDirection) 
    {
        int x = std::get<0>( GridCoords );
        int y = std::get<1>( GridCoords );
        if ( y % 2 == 0 ) { // Odd row
            switch ( AttackDirection ) {
            case 1: // Down-Right
                x += 1;
                y += 1;
                break;
            case 2: // Down-Left
                y += 1;
                break;
            case 4: // LEFT
                x -= 1;
                break;
            case 8: // UP-Left
                y -= 1;
                break;
            case 16: // UP-Right
                x += 1;
                y -= 1;
                break;
            case 32: // Right
                x += 1;
                break;
            default:
                std::cerr << "Invalid attack direction!" << std::endl;
            }
        }
        else { // Even row
        switch ( AttackDirection ) {
        case 1: //Down-Right
            y += 1;
            break;
        case 2: // Down-Left
            x -= 1;
            y += 1;
            break;
        case 4: // LEFT
            x -= 1;
            break;
        case 8: // UP-Left
            x -= 1;
            y -= 1;
            break;
        case 16: // UP-Right
            y -= 1;
            break;
        case 32: // Right
            x += 1;
            break;
        default:
            std::cerr << "Invalid attack direction!" << std::endl;
        }
        return std::make_tuple( x, y );
        }
        if ( x < 0 || x >= 11 || y < 0 || y >= 9 ) {
            std::cerr << "Coordinates out of bounds after attack!" << std::endl;
            return std::make_tuple( -1, -1 ); // Invalid coordinates
        }
        return std::make_tuple( x, y );
    }

    int coordinates_to_grid_id( int x, int y )
    {
        if ( x < 0 || x >= 11 || y < 0 || y >= 9 ) {
            std::cerr << "Coordinates out of bounds!" << std::endl;
            return -1; // Invalid grid ID
        }
        return ( y * 11 + x + 1 ); 
    }

    int apply_attack_to_grid( int GridID, int AttackDirection )
    {
        auto coords = grid_id_to_coordinates( GridID );
        auto new_coords = apply_attack_to_coordinates( coords, AttackDirection );
        return coordinates_to_grid_id( std::get<0>( new_coords ), std::get<1>( new_coords ) );
    }

    std::vector<torch::Tensor> predict( BattleLSTM & model, const torch::Tensor & input )
    {
       model->eval(); // Set the model to evaluation mode
       torch::NoGradGuard no_grad; 
       auto output = model->forward( input );
       return output;
    }

    // Converts a vector of action indices (from NN output) to Battle::Actions
    Battle::Actions predict_action( const Battle::Unit & currentUnit, const Battle::Arena & arena ) // , const Battle::Arena & arena 
    {
        //BattleLSTM model = getModelByColor(currentUnit.GetColor()); //TODO
        BattleLSTM & model = *g_model1; // Placeholder for actual model retrieval logic
        // Null pointer protection for model
        
        if ( !model) {
            std::cerr << "Error: Neural network model (g_model1) is not initialized!" << std::endl;
            return {};
        }
        
        std::vector<torch::Tensor> & nn_output = model ->forward( NNAI::prepareBattleLSTMInput( arena, currentUnit ) ); // Get the NN output

        Battle::Actions actions;

        if ( nn_output.empty() )
            return actions;

        int actionType = nn_output[0].argmax( 1 ).item<int>(); // Action type index from NN output (0: SKIP, 1: MOVE, 2: ATTACK, 3: SPELLCAST, 4: RETREAT, 5: SURRENDER)
        int positionNum = nn_output[1].argmax( 1 ).item<int>(); // Position index for MOVE, ATTACK, SPELLCAST
        int attack_direction = nn_output[3].argmax( 1 ).item<int>(); // Direction index for ATTACK (0-6)
        attack_direction = ( attack_direction >= 6 ? -1 : 1 << attack_direction ); // Convert to actual direction (1,2,4,8,16,32) or -1 for archery
        int currentUnitUID = currentUnit.GetUID(); // Current unit UID
        int attackTargetPositon = apply_attack_to_grid(positionNum, attack_direction);
        int targetUnitId = -1; // TODO Get the target unit ID at the specified position
        auto cell = arena.GetBoard()->GetCell( positionNum );
        if ( cell ) {
            auto unit = cell->GetUnit();
            if ( unit ) {
                targetUnitId = unit->GetID();
            }
        }

        actionType = 1;

        std::cout << std::endl
                  << "Action Type: "<< actionType
                  << ", Position Num: "<< positionNum
                  << ", Attack Direction: "<< attack_direction 
                  << ", Current Unit UID: "<< currentUnitUID 
                  << ", Target Unit ID: " << targetUnitId
                  << ", Attack Target Position: " << attackTargetPositon << std::endl;



        switch ( actionType ) {
        case 0:
            // MOVE: [CommandType, unitUID, targetCell]
            actions.emplace_back( Battle::Command::MOVE, currentUnitUID, positionNum );
            break;
        case 1:
            // ATTACK: [CommandType, unitUID, targetUnitUID, moveTargetIdx, attackTargetIdx, attackDirection]
            actions.emplace_back( Battle::Command::ATTACK, currentUnitUID, targetUnitId, positionNum, attackTargetPositon, attack_direction );
            // TODO
            break;
        case 2:
            // SPELLCAST: [CommandType, spellID, targetCell]
            //actions.emplace_back( Battle::Command::SPELLCAST, nn_output[1], nn_output[2] );
            actions.emplace_back( Battle::Command::SKIP, currentUnitUID );
            break;
        case 3:
            actions.emplace_back( Battle::Command::RETREAT );
            break;
        case 4:
            actions.emplace_back( Battle::Command::SURRENDER );
            break;
        case 5:
            // SKIP: [CommandType, unitUID]
            actions.emplace_back( Battle::Command::SKIP, currentUnitUID );
            break;
        default:
            // Unknown command, fallback to SKIP
            actions.emplace_back( Battle::Command::SKIP, currentUnitUID );
            break;
        }

        return actions;
    }

    // Normalize a value to the range [0, 1]
    inline float normalize( float value, float min, float max )
    {
        return ( value - min ) / ( max - min );
    }

    // Extract features for a single unit
    std::vector<float> extractUnitFeatures( const Battle::Unit & unit, const Battle::Arena & arena )
    {
        std::vector<float> features;
        
        features.push_back( unit.GetUID() ); // Unique ID
        features.push_back( normalize(unit.GetHitPoints(), 0, 500 ) ); // Normalize HP
        features.push_back( normalize(unit.GetSpeed( false, true ), 0, 20 ) ); // Normalize speed
        features.push_back( normalize(unit.GetAttack(), 0, 100 ) ); // Normalize attack
        features.push_back( normalize(unit.GetDefense(), 0, 100 ) ); // Normalize defense
        features.push_back( unit.isFlying() ? 1.0f : 0.0f ); // Is flying
        features.push_back( unit.isArchers() ? 1.0f : 0.0f ); // Is archer
        features.push_back( unit.isHandFighting() ? 1.0f : 0.0f ); // Is hand fighting
        features.push_back( unit.isWide() ? 1.0f : 0.0f ); // Is wide
        features.push_back( unit.isAffectedByMorale() ? 1.0f : 0.0f ); // Affected by morale
        features.push_back( unit.isImmovable() ? 1.0f : 0.0f ); // Is immovable
        features.push_back( unit.GetMorale() ); // Morale
        features.push_back( unit.GetLuck() ); // Luck
        features.push_back( unit.GetColor() ); // Ally or enemy color
        features.push_back(normalize( unit.GetPosition().GetHead()->GetIndex(), 0, 100 )); // Position (normalized by battlefield size)
        
        return features;
    }

    torch::Tensor prepareBattleLSTMInput( const Battle::Arena & arena, const Battle::Unit & currentUnit )
    {
        // Get all units for both sides
        int _myColor = currentUnit.GetCurrentColor();
        
        const Battle::Units enemies( arena.getEnemyForce( _myColor ).getUnits(), Battle::Units::REMOVE_INVALID_UNITS_AND_SPECIFIED_UNIT, &currentUnit );
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
            return torch::empty( { 1, 0, 0 } );

        const int seq_len = static_cast<int>( featuresList.size() );
        const int input_size = static_cast<int>( featuresList[0].size() );

        torch::Tensor input = torch::zeros( { 1, seq_len, input_size }, torch::kFloat32 );
        for ( int i = 0; i < seq_len; ++i ) {
            for ( int j = 0; j < input_size; ++j ) {
                input[0][i][j] = featuresList[i][j];
            }
        }
        
        return input;
    }


    void trainingGameLoop( bool isFirstGameRun, bool isProbablyDemoVersion, int training_loops )
    {
        fheroes2::GameMode result = fheroes2::GameMode::NEW_BATTLE_ONLY;

        bool exit = false;

        while ( !exit && training_loops >= 0 ) {
            training_loops--; // Will loop n times, then exit
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

        // We are quitting the game, so fade-out the screen.
        fheroes2::fadeOutDisplay();
    }

}
#include "ostream"
#include <string>

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

}