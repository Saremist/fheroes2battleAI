#include "NN_ai.h"

#include <iostream>
#include <torch/torch.h>
#include <filesystem>
#include "battle.h"
#include "battle_command.h"
//#include "battle_action.h"
#include "battle_arena.h"
#include "battle_army.h"
#include <algorithm> // For std::reverse

#include <tuple>

#include "game.h"
#include "ui_tool.h"

namespace NNAI
{
    // Global model pointers for each color
    std::shared_ptr<BattleLSTM> g_model_blue = nullptr;
    std::shared_ptr<BattleLSTM> g_model_green = nullptr;
    std::shared_ptr<BattleLSTM> g_model_red = nullptr;
    std::shared_ptr<BattleLSTM> g_model_yellow = nullptr;
    std::shared_ptr<BattleLSTM> g_model_orange = nullptr;
    std::shared_ptr<BattleLSTM> g_model_purple = nullptr;
    const bool isTraining = false; // Defines if post battle dialog will open or the training loop will continue
    const int TrainingLoopsCount = 10;

    torch::Device device( torch::kCPU );

    void initializeGlobalModels()
    {
        loadModel( NNAI::g_model_blue, "model_blue.pt" );
        loadModel( NNAI::g_model_green, "model_green.pt" );
        loadModel( NNAI::g_model_red, "model_red.pt" );
        loadModel( NNAI::g_model_yellow, "model_yellow.pt" );
        loadModel( NNAI::g_model_orange, "model_orange.pt" );
        loadModel( NNAI::g_model_purple, "model_purple.pt" );
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

    
     std::shared_ptr<BattleLSTM> getModelByColor( int color )
       {
         switch ( color ) {
           case 0x01: // BLUE
               return g_model_blue;
           case 0x02: // GREEN
               return g_model_green;
           case 0x04: // RED
               return g_model_red;
           case 0x08: // YELLOW
               return g_model_yellow;
           case 0x10:// ORANGE
               return g_model_orange;
           case 0x20: // PURPLE
               return g_model_purple;
           default:
               std::cerr << "Warning: Unrecognized color " << color << ". Returning default model." << std::endl;
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

    std::vector<torch::Tensor> predict( BattleLSTM & model, const torch::Tensor & input )
    {
       model->eval(); // Set the model to evaluation mode
       torch::NoGradGuard no_grad; 
       auto output = model->forward( input );
       return output;
    }


    // Converts a vector of action indices (from NN output) to Battle::Actions
    Battle::Actions predict_action( const Battle::Unit & currentUnit, Battle::Arena & arena ) // , const Battle::Arena & arena 
    {
        BattleLSTM & model = * getModelByColor(currentUnit.GetColor());
        //BattleLSTM & model = *g_model1; // Placeholder for actual model retrieval logic
        
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


        //int attackTargetPositon = apply_attack_to_grid(positionNum, attack_direction);
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
            actionType = 5; // SKIP move to already ocupied slot
            std::cout << "Already at " << currentUnit.GetPosition().GetHead() << ". Defaulting to SKIP." << std::endl;
        }
        
        
        std::cout << std::endl
                  << "Action Type: "<< actionType
                  << ", Position Num: "<< positionNum
                  << ", Attack Direction: "<< attack_direction 
                  << ", Current Unit UID: "<< currentUnitUID 
                  << ", Target unit UID: " << targetUnitUID
                  << ", Attack Target Position: " << attackTargetPositon << std::endl;


        if (actionType == 1 
            && !CheckAttackParameters( &currentUnit, arena.GetBoard()->GetCell(attackTargetPositon)->GetUnit(), positionNum, attackTargetPositon, attack_direction))
        {
            actionType = 5; // SKIP attack to illegal position ends up as SKIP
            std::cout << "Attack to position " << attackTargetPositon << " is not available for unit " << currentUnitUID << ". Defaulting to SKIP." << std::endl;
        }               
        else if ( actionType == 0 && !CheckMoveParameters( &currentUnit, positionNum ) ) {
            actionType = 5; // SKIP move to illegal positon ends up as SKIP
            std::cout << "Position " << positionNum << " is not available for unit " << currentUnitUID << ". Defaulting to SKIP." << std::endl;
        }
                            

        switch ( actionType ) {
        case 0:
            // MOVE: [CommandType, unitUID, targetCell
            std::cout << "Moving unit " << currentUnitUID << " to position " << positionNum << std::endl;
            actions.emplace_back( Battle::Command::MOVE, currentUnitUID, positionNum );
            break;
        case 1:
            // ATTACK: [CommandType, unitUID, targetUnitUID, moveTargetIdx, attackTargetIdx, attackDirection]
            actions.emplace_back( Battle::Command::ATTACK, currentUnitUID, targetUnitUID, positionNum, attackTargetPositon, attack_direction );
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

void PrintUnitInfo( const Battle::Unit & unit ) 
{
    std::cout << "Unit Name: " << unit.GetName() << ", Unit Id:" << unit.GetID() << ", Unit UID: "<< unit.GetUID() << ", Count: " << unit.GetCount()
              << ", Position: " << unit.GetPosition().GetHead()->GetIndex() << ", Shooting: " << unit.GetShots() << ", speed: " << unit.GetSpeed()
              << " HitPoints: " << unit.GetHitPointsLeft() << std::endl;
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