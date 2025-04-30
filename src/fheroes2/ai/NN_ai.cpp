#include "NN_ai.h"

#include <iostream>
#include <torch/torch.h>

namespace NNAI
{
   torch::Device device( torch::kCPU );

   void createAndSaveModel( const std::string & model_path )
   {
       int64_t input_size = 10; // Adjust based on your input features
       int64_t hidden_size = 128;
       int64_t output_size = 5; // Adjust based on your output actions
       int64_t num_layers = 1;

       try {
           BattleLSTM model( input_size, hidden_size, output_size, num_layers );
           // Save the model
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

           // Self-play loop (e.g., simulate N games per epoch)
           for ( int i = 0; i < NUM_SELF_PLAY_GAMES; ++i ) {
               // Simulate a game between current model and opponent (could be same model)
               std::vector<torch::Tensor> states, actions, rewards;

               //simulateSelfPlayGame( model, states, actions, rewards, device );

               // Convert vectors to tensors (batch format)
               auto state_batch = torch::stack( states ).to( device );
               auto action_batch = torch::stack( actions ).to( device );
               auto reward_batch = torch::stack( rewards ).to( device );

               optimizer.zero_grad();

               // Get model output
               auto logits = model->forward( state_batch );

               // Calculate loss — here using policy gradient (REINFORCE) as an example
               auto log_probs = torch::nn::functional::log_softmax( logits, /*dim=*/1 );
               auto selected_log_probs = log_probs.gather( 1, action_batch.unsqueeze( 1 ) ).squeeze();

               auto loss = -( selected_log_probs * reward_batch ).mean();

               loss.backward();
               optimizer.step();

               total_loss += loss.item<double>();
               ++game_count;
           }

           std::cout << "Epoch [" << ( epoch + 1 ) << "/" << num_epochs << "], Avg Loss: " << ( total_loss / game_count ) << std::endl;
       }
   }

   BattleLSTM NNAI::createModel( int64_t input_size, int64_t hidden_size, int64_t output_size, int64_t num_layers )
   {
       return BattleLSTM( input_size, hidden_size, output_size, num_layers );
   }

   void NNAI::saveModel( const BattleLSTM & model, const std::string & model_path )
   {
       try {
           torch::save( model, model_path );
           std::cout << "Model saved to " << model_path << std::endl;
       }
       catch ( const std::exception & e ) {
           std::cerr << "Error saving the model: " << e.what() << std::endl;
       }
   }

   BattleLSTM NNAI::loadModel( const std::string & model_path )
   {
       BattleLSTM model;
       try {
           torch::load( model, model_path );
           std::cout << "Model loaded from " << model_path << std::endl;
       }
       catch ( const std::exception & e ) {
           std::cerr << "Error loading the model: " << e.what() << std::endl;
       }
       return model;
   }

    torch::Tensor NNAI::predict( BattleLSTM & model, const torch::Tensor & input )
    {
       model->eval(); // Set the model to evaluation mode
       torch::NoGradGuard no_grad; // Disable gradient computation for inference

       // Perform forward pass
       auto output = model->forward( input );

       // Return the output tensor
       return output;
    }

#include <cmath>
#include <vector>

    // Normalize a value to the range [0, 1]
    inline float normalize( float value, float min, float max )
    {
        return ( value - min ) / ( max - min );
    }

    // Extract features for a single unit
    std::vector<float> extractUnitFeatures( const Battle::Unit & unit, const Battle::Arena & arena )
    {
        std::vector<float> features;

        // Example features
        features.push_back( normalize( unit.GetHitPoints(), 0, 1000 ) ); // Normalize HP
        features.push_back( normalize( unit.GetSpeed( false, true ), 0, 20 ) ); // Normalize speed
        features.push_back( normalize( unit.GetAttack(), 0, 100 ) ); // Normalize attack
        features.push_back( normalize( unit.GetDefense(), 0, 100 ) ); // Normalize defense
        features.push_back( unit.isFlying() ? 1.0f : 0.0f ); // Is flying
        features.push_back( unit.isArchers() ? 1.0f : 0.0f ); // Is archer

        // Position (normalized by battlefield size)
        const Battle::Position & pos = unit.GetPosition();
        if ( pos.GetHead() ) {
            features.push_back( normalize( pos.GetHead()->GetIndex(), 0, Battle::Board::sizeInCells ) );
        }
        else {
            features.push_back( 0.0f ); // Default value if no position
        }

        // Add more features as needed...

        return features;
    }

}