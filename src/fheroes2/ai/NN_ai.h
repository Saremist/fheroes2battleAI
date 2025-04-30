#ifndef FHEROES2_AI_NN_AI_H
#define FHEROES2_AI_NN_AI_H



#include <string>
#include <torch/torch.h>

namespace NNAI
{
    struct BattleLSTMImpl : torch::nn::Module
    {
        torch::nn::LSTM lstm_layer{ nullptr };
        torch::nn::Linear fc{ nullptr };

        // Default constructor
        BattleLSTMImpl()
            : lstm_layer( torch::nn::LSTMOptions( 10, 128 ).num_layers( 1 ).batch_first( true ) )
            , // Default values
            fc( 128, 5 ) // Default values
        {
            register_module( "lstm_layer", lstm_layer );
            register_module( "fc", fc );
        }

        // Parameterized constructor
        BattleLSTMImpl( int64_t input_size, int64_t hidden_size, int64_t output_size, int64_t num_layers )
            : lstm_layer( torch::nn::LSTMOptions( input_size, hidden_size ).num_layers( num_layers ).batch_first( true ) )
            , fc( hidden_size, output_size )
        {
            register_module( "lstm_layer", lstm_layer );
            register_module( "fc", fc );
        }

        torch::Tensor forward( torch::Tensor x )
        {
            auto h0 = torch::zeros( { lstm_layer->options.num_layers(), x.size( 0 ), lstm_layer->options.hidden_size() } );
            auto c0 = torch::zeros( { lstm_layer->options.num_layers(), x.size( 0 ), lstm_layer->options.hidden_size() } );

            auto lstm_out = std::get<0>( lstm_layer( x, std::make_tuple( h0, c0 ) ) );
            auto last_timestep = lstm_out.index( { -1 } );
            return fc( last_timestep );
        }
    };
    TORCH_MODULE( BattleLSTM );

    // Declaration of createAndSaveModel
    void createAndSaveModel( const std::string & model_path );
    BattleLSTM createModel( int64_t input_size, int64_t hidden_size, int64_t output_size, int64_t num_layers );
    void saveModel( const BattleLSTM & model, const std::string & model_path );
    BattleLSTM loadModel( const std::string & model_path );
    void trainModel( BattleLSTM & model, int64_t num_epochs, double learning_rate, torch::Device device );
    torch::Tensor predict( BattleLSTM & model, const torch::Tensor & input );
    torch::Tensor preprocessInput( const std::vector<float> & raw_data );
    //double evaluateModel( const BattleLSTM & model, torch::data::DataLoader<torch::data::Example<>> & test_loader );

}

#endif // FHEROES2_AI_NN_AI_H