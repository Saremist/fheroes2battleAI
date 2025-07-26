/***************************************************************************
 *   fheroes2: https://github.com/ihhub/fheroes2                           *
 *   Copyright (C) 2019 - 2024                                             *
 *                                                                         *
 *   Free Heroes2 Engine: http://sourceforge.net/projects/fheroes2         *
 *   Copyright (C) 2009 by Andrey Afletdinov <fheroes2@gmail.com>          *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <vector>

// Managing compiler warnings for SDL headers
#if defined( __GNUC__ )
#pragma GCC diagnostic push

#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif

#include <SDL_events.h>
#include <SDL_main.h> // IWYU pragma: keep
#include <SDL_mouse.h>

// Managing compiler warnings for SDL headers
#if defined( __GNUC__ )
#pragma GCC diagnostic pop
#endif

#if defined( _WIN32 )
#include <cassert>
#endif

#include "agg.h"
#include "agg_image.h"
#include "audio_manager.h"
#include "core.h"
#include "cursor.h"
#include "dir.h"
#include "embedded_image.h"
#include "exception.h"
#include "game.h"
#include "game_logo.h"
#include "game_video.h"
#include "game_video_type.h"
#include "h2d.h"
#include "icn.h"
#include "image.h"
#include "image_palette.h"
#include "localevent.h"
#include "logging.h"
#include "render_processor.h"
#include "screen.h"
#include "settings.h"
#include "system.h"
#include "timing.h"
#include "ui_tool.h"
#include "zzlib.h"

#pragma warning( disable : 4996 )

namespace
{
    std::string GetCaption()
    {
        return std::string( "fheroes2 engine, version: " + Settings::GetVersion() );
    }

    void ReadConfigs()
    {
        const std::string configurationFileName( Settings::configFileName );
        const std::string confFile = Settings::GetLastFile( "", configurationFileName );

        Settings & conf = Settings::Get();
        if ( System::IsFile( confFile ) && conf.Read( confFile ) ) {
            LocalEvent::Get().SetControllerPointerSpeed( conf.controllerPointerSpeed() );
        }
        else {
            conf.Save( configurationFileName );

            // Fullscreen mode can be enabled by default for some devices, we need to forcibly
            // synchronize reality with the default config if config file was not read
            conf.setFullScreen( conf.FullScreen() );
        }
    }

    void InitConfigDir()
    {
        const std::string configDir = System::GetConfigDirectory( "fheroes2" );

        System::MakeDirectory( configDir );
    }

    void InitDataDir()
    {
        const std::string dataDir = System::GetDataDirectory( "fheroes2" );

        if ( dataDir.empty() ) {
            return;
        }

        const std::string dataFiles = System::concatPath( dataDir, "files" );
        const std::string dataFilesSave = System::concatPath( dataFiles, "save" );

        // This call will also create dataDir and dataFiles
        System::MakeDirectory( dataFilesSave );
    }

    void displayMissingResourceWindow()
    {
        fheroes2::Display & display = fheroes2::Display::instance();
        const fheroes2::Image & image = Compression::CreateImageFromZlib( 290, 190, errorMessage, sizeof( errorMessage ), false );

        display.fill( 0 );
        fheroes2::Resize( image, display );

        display.render();

        LocalEvent & le = LocalEvent::Get();

        // Display the message for 5 seconds so that the user sees it enough and not immediately closes without reading properly.
        const fheroes2::Time timer;

        bool closeWindow = false;

        while ( le.HandleEvents( true, true ) ) {
            if ( closeWindow && timer.getS() >= 5 ) {
                break;
            }

            if ( le.isAnyKeyPressed() || le.MouseClickLeft() ) {
                closeWindow = true;
            }
        }
    }

    class DisplayInitializer
    {
    public:
        DisplayInitializer()
        {
            const Settings & conf = Settings::Get();

            fheroes2::Display & display = fheroes2::Display::instance();
            fheroes2::ResolutionInfo bestResolution{ conf.currentResolutionInfo() };

            if ( conf.isFirstGameRun() && System::isHandheldDevice() ) {
                // We do not show resolution dialog for first run on handheld devices. In this case it is wise to set 'widest' resolution by default.
                const std::vector<fheroes2::ResolutionInfo> resolutions = fheroes2::engine().getAvailableResolutions();

                for ( const fheroes2::ResolutionInfo & info : resolutions ) {
                    if ( info.gameWidth > bestResolution.gameWidth && info.gameHeight == bestResolution.gameHeight ) {
                        bestResolution = info;
                    }
                }
            }

            display.setResolution( bestResolution );

            fheroes2::engine().setTitle( GetCaption() );

            SDL_ShowCursor( SDL_DISABLE ); // hide system cursor

            fheroes2::RenderProcessor & renderProcessor = fheroes2::RenderProcessor::instance();

            display.subscribe( [&renderProcessor]( std::vector<uint8_t> & palette ) { return renderProcessor.preRenderAction( palette ); },
                               [&renderProcessor]() { renderProcessor.postRenderAction(); } );

            // Initialize system info renderer.
            _systemInfoRenderer = std::make_unique<fheroes2::SystemInfoRenderer>();

            renderProcessor.registerRenderers( [sysInfoRenderer = _systemInfoRenderer.get()]() { sysInfoRenderer->preRender(); },
                                               [sysInfoRenderer = _systemInfoRenderer.get()]() { sysInfoRenderer->postRender(); } );
            renderProcessor.startColorCycling();

            // Update mouse cursor when switching between software emulation and OS mouse modes.
            fheroes2::cursor().registerUpdater( Cursor::Refresh );

#if !defined( MACOS_APP_BUNDLE )
            const fheroes2::Image & appIcon = Compression::CreateImageFromZlib( 32, 32, iconImage, sizeof( iconImage ), true );
            fheroes2::engine().setIcon( appIcon );
#endif
        }

        DisplayInitializer( const DisplayInitializer & ) = delete;
        DisplayInitializer & operator=( const DisplayInitializer & ) = delete;

        ~DisplayInitializer()
        {
            fheroes2::RenderProcessor::instance().unregisterRenderers();

            fheroes2::Display & display = fheroes2::Display::instance();
            display.subscribe( {}, {} );
            display.release();
        }

    private:
        // This member must not be initialized before Display.
        std::unique_ptr<fheroes2::SystemInfoRenderer> _systemInfoRenderer;
    };

    class DataInitializer
    {
    public:
        DataInitializer()
        {
            const fheroes2::ScreenPaletteRestorer screenRestorer;

            try {
                _aggInitializer.reset( new AGG::AGGInitializer );

                _h2dInitializer.reset( new fheroes2::h2d::H2DInitializer );

                // Verify that the font is present and it is not corrupted.
                fheroes2::AGG::GetICN( ICN::FONT, 0 );
            }
            catch ( ... ) {
                displayMissingResourceWindow();

                throw;
            }
        }

        DataInitializer( const DataInitializer & ) = delete;
        DataInitializer & operator=( const DataInitializer & ) = delete;
        ~DataInitializer() = default;

        const std::string & getOriginalAGGFilePath() const
        {
            return _aggInitializer->getOriginalAGGFilePath();
        }

        const std::string & getExpansionAGGFilePath() const
        {
            return _aggInitializer->getExpansionAGGFilePath();
        }

    private:
        std::unique_ptr<AGG::AGGInitializer> _aggInitializer;
        std::unique_ptr<fheroes2::h2d::H2DInitializer> _h2dInitializer;
    };

    // This function checks for a possible situation when a user uses a demo version
    // of the game. There is no 100% certain way to detect this, so assumptions are made.
    bool isProbablyDemoVersion()
    {
        if ( Settings::Get().isPriceOfLoyaltySupported() ) {
            return false;
        }

        // The demo version of the game only has 1 map.
        const ListFiles maps = Settings::FindFiles( "maps", ".mp2", false );
        return maps.size() == 1;
    }
}

#include "NN_ai.h";
/*
Main Training loop for the game.
It chcages default game loop to run training games.
This will instatnly loop through the game and collect data for training n times specified by NNAI::TrainingLoopsCount.
This is used to train the neural network models for the game.
It is not intended to be used by the end user.

Modified from the original fheroes2::Game::mainGameLoop() by
Milan Wróblewski for the purpose of Engineer thesis.
*/
int NNAI::training_main( int argc, char ** argv, int64_t num_epochs, double learning_rate, torch::Device device, int64_t NUM_SELF_PLAY_GAMES )
{
#if defined( _WIN32 )
    assert( argc == __argc );

    argv = __argv;
#else
    (void)argc;
#endif
    try {
        const fheroes2::HardwareInitializer hardwareInitializer;
        Logging::InitLog();

        COUT( GetCaption() )

        Settings & conf = Settings::Get();
        conf.SetProgramPath( argv[0] );

        InitConfigDir();
        InitDataDir();
        ReadConfigs();

        std::set<fheroes2::SystemInitializationComponent> coreComponents{ fheroes2::SystemInitializationComponent::Audio,
                                                                          fheroes2::SystemInitializationComponent::Video };

#if defined( TARGET_PS_VITA ) || defined( TARGET_NINTENDO_SWITCH )
        coreComponents.emplace( fheroes2::SystemInitializationComponent::GameController );
#endif

        const fheroes2::CoreInitializer coreInitializer( coreComponents );

        DEBUG_LOG( DBG_GAME, DBG_INFO, conf.String() )

        const DisplayInitializer displayInitializer;
        const DataInitializer dataInitializer;

        ListFiles midiSoundFonts;

        midiSoundFonts.Append( Settings::FindFiles( System::concatPath( "files", "soundfonts" ), ".sf2", false ) );
        midiSoundFonts.Append( Settings::FindFiles( System::concatPath( "files", "soundfonts" ), ".sf3", false ) );

#ifdef WITH_DEBUG
        for ( const std::string & file : midiSoundFonts ) {
            DEBUG_LOG( DBG_GAME, DBG_INFO, "MIDI SoundFont to load: " << file )
        }
#endif

        const AudioManager::AudioInitializer audioInitializer( dataInitializer.getOriginalAGGFilePath(), dataInitializer.getExpansionAGGFilePath(), midiSoundFonts );

        // Load palette.
        fheroes2::setGamePalette( AGG::getDataFromAggFile( "KB.PAL" ) );
        fheroes2::Display::instance().changePalette( nullptr, true );

        // init game data
        Game::Init();

        conf.setGameLanguage( conf.getGameLanguage() );

        try {
            const CursorRestorer cursorRestorer( true, Cursor::POINTER );

            // Traing Setup

            double total_elapsed_seconds = 0.0;

            for ( int64_t epoch = 0; epoch < num_epochs; ++epoch ) {
                auto epoch_start = std::chrono::steady_clock::now(); // CHRONO

                auto selection = NNAI::SelectRandomModels();
                BattleLSTM & model1 = std::get<0>( selection );
                std::string name1 = std::get<1>( selection );
                BattleLSTM & model2 = std::get<2>( selection );
                std::string name2 = std::get<3>( selection );

                model1->train();
                model2->train();

                // Assign models to global pointers used during game
                NNAI::g_model1 = std::make_shared<NNAI::BattleLSTM>( model1 );
                NNAI::g_model2 = std::make_shared<NNAI::BattleLSTM>( model2 );

                torch::optim::Adam optimizer1( model1->parameters(), torch::optim::AdamOptions( learning_rate ) );
                torch::optim::Adam optimizer2( model2->parameters(), torch::optim::AdamOptions( learning_rate ) );

                double total_loss1 = 0.0, total_loss2 = 0.0;
                int game_count = 0;
                double epoch_total_reward1 = 0.0, epoch_total_reward2 = 0.0; // Total rewards for this epoch

                for ( int i = 0; i < NUM_SELF_PLAY_GAMES; ++i ) { // C
                    int copy_argc = argc;
                    std::vector<std::string> argv_strings( argc );
                    for ( int i = 0; i < argc; ++i ) {
                        argv_strings[i] = argv[i];
                    }
                    std::vector<char *> copy_argv_vec( argc );
                    for ( int i = 0; i < argc; ++i ) {
                        copy_argv_vec[i] = argv_strings[i].data();
                    }
                    char ** copy_argv = copy_argv_vec.data();

                    // === Data containers for the current game ===
                    std::vector<torch::Tensor> states1, states2;
                    std::vector<std::vector<torch::Tensor>> actions1( 3 ), actions2( 3 );
                    std::vector<torch::Tensor> rewards1, rewards2;

                    NNAI::g_states1 = &states1;
                    NNAI::g_actions1 = &actions1;
                    NNAI::g_rewards1 = &rewards1;
                    NNAI::g_states2 = &states2;
                    NNAI::g_actions2 = &actions2;
                    NNAI::g_rewards2 = &rewards2;

                    NNAI::trainingGameLoop( false, isProbablyDemoVersion() ); // GAME LOOP

                    if ( states1.empty() || rewards1.empty() || actions1[0].empty() || states2.empty() || rewards2.empty() || actions2[0].empty() ) {
                        std::cerr << "Warning: Empty game data. Skipping game " << i << std::endl;
                        continue;
                    }

                    // Calculate sum of rewards for both players
                    float sum_rewards1 = 0.0f;
                    for ( const auto & r : rewards1 ) {
                        sum_rewards1 += r.cpu().item<float>();
                    }
                    float sum_rewards2 = 0.0f;
                    for ( const auto & r : rewards2 ) {
                        sum_rewards2 += r.cpu().item<float>();
                    }

                    // Accumulate total rewards for the epoch
                    epoch_total_reward1 += sum_rewards1;
                    epoch_total_reward2 += sum_rewards2;

                    // === Model 1 Training ===
                    {
                        torch::Tensor state_batch1 = torch::stack( states1 ).to( device );
                        torch::Tensor reward_batch1 = torch::stack( rewards1 ).to( device ).view( { -1 } );

                        std::vector<torch::Tensor> action_batches1;
                        for ( int h = 0; h < 3; ++h )
                            action_batches1.push_back( torch::stack( actions1[h] ).to( device ) );

                        int64_t T = std::min( { state_batch1.size( 0 ), reward_batch1.size( 0 ), action_batches1[0].size( 0 ) } );
                        state_batch1 = state_batch1.slice( 0, 0, T );
                        reward_batch1 = reward_batch1.slice( 0, 0, T );
                        for ( int h = 0; h < 3; ++h )
                            action_batches1[h] = action_batches1[h].slice( 0, 0, T );

                        optimizer1.zero_grad();
                        auto logits1 = model1->forward( state_batch1 );
                        if ( logits1.size() != 3 || action_batches1.size() != 3 ) {
                            std::cerr << "Warning: Invalid logits or action batches for model1\n";
                            continue;
                        }

                        auto reward_mean = reward_batch1.mean().detach();
                        auto reward_std = torch::clamp( reward_batch1.std().detach(), 1e-8 );
                        if ( reward_batch1.size( 0 ) <= 1 || reward_std.item<float>() < 1e-8 ) {
                            reward_std = torch::ones( {}, reward_std.options() ) * 1.0; // or skip normalization
                        }
                        auto norm_rewards = ( reward_batch1 - reward_mean ) / reward_std;

                        torch::Tensor loss1 = torch::zeros( {}, torch::TensorOptions().dtype( torch::kFloat32 ).device( device ) );
                        for ( int h = 0; h < 3; ++h ) {
                            auto log_prob = torch::nn::functional::log_softmax( logits1[h], 1 );
                            auto selected_log_prob = log_prob.gather( 1, action_batches1[h].unsqueeze( 1 ) );
                            loss1 -= ( selected_log_prob.squeeze() * norm_rewards ).mean();
                        }

                        loss1.backward();
                        optimizer1.step();
                        total_loss1 += loss1.cpu().item<double>();
                    }

                    // === Model 2 Training ===
                    {
                        torch::Tensor state_batch2 = torch::stack( states2 ).to( device );
                        torch::Tensor reward_batch2 = torch::stack( rewards2 ).to( device ).view( { -1 } );

                        std::vector<torch::Tensor> action_batches2;
                        for ( int h = 0; h < 3; ++h )
                            action_batches2.push_back( torch::stack( actions2[h] ).to( device ) );

                        int64_t T = std::min( { state_batch2.size( 0 ), reward_batch2.size( 0 ), action_batches2[0].size( 0 ) } );
                        state_batch2 = state_batch2.slice( 0, 0, T );
                        reward_batch2 = reward_batch2.slice( 0, 0, T );
                        for ( int h = 0; h < 3; ++h )
                            action_batches2[h] = action_batches2[h].slice( 0, 0, T );

                        optimizer2.zero_grad();
                        auto logits2 = model2->forward( state_batch2 );
                        if ( logits2.size() != 3 || action_batches2.size() != 3 ) {
                            std::cerr << "Warning: Invalid logits or action batches for model2\n";
                            continue;
                        }

                        auto reward_mean = reward_batch2.mean().detach();
                        auto reward_std = torch::clamp( reward_batch2.std().detach(), 1e-8 );

                        if ( reward_batch2.size( 0 ) <= 1 || reward_std.item<float>() < 1e-8 ) {
                            reward_std = torch::ones( {}, reward_std.options() ) * 1.0; // or skip normalization
                        }
                        auto norm_rewards = ( reward_batch2 - reward_mean ) / reward_std;

                        torch::Tensor loss2 = torch::zeros( {}, torch::TensorOptions().dtype( torch::kFloat32 ).device( device ) );
                        for ( int h = 0; h < 3; ++h ) {
                            auto log_prob = torch::nn::functional::log_softmax( logits2[h], 1 );
                            auto selected_log_prob = log_prob.gather( 1, action_batches2[h].unsqueeze( 1 ) );
                            loss2 -= ( selected_log_prob.squeeze() * norm_rewards ).mean();
                        }

                        loss2.backward();
                        optimizer2.step();
                        total_loss2 += loss2.cpu().item<double>();
                    }

                    ++game_count;
                }
                auto epoch_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;
                total_elapsed_seconds += epoch_duration.count();
                double avg_epoch_time = total_elapsed_seconds / static_cast<double>( epoch + 1 );
                int64_t remaining_epochs = num_epochs - ( epoch + 1 );
                double estimated_remaining_time = avg_epoch_time * remaining_epochs;
                double games_per_second = ( epoch_duration.count() > 0.0 ) ? ( game_count / epoch_duration.count() ) : 0.0;

                int percent_complete = static_cast<int>( ( ( epoch + 1.0 ) / num_epochs ) * 100.0 );

                // Format time
                auto format_seconds = []( double seconds ) -> std::string {
                    int hrs = static_cast<int>( seconds ) / 3600;
                    int mins = ( static_cast<int>( seconds ) % 3600 ) / 60;
                    int secs = static_cast<int>( seconds ) % 60;
                    char buffer[64];
                    std::snprintf( buffer, sizeof( buffer ), "%02d:%02d:%02d", hrs, mins, secs );
                    return std::string( buffer );
                };

                // === Epoch summary ===
                std::string epochSummary = "Epoch " + std::to_string( epoch + 1 ) + "/" + std::to_string( num_epochs ) + " (" + std::to_string( percent_complete ) + "%)"
                                           + " | Time: " + format_seconds( epoch_duration.count() ) + " | ETA: " + format_seconds( estimated_remaining_time )
                                           + " | GPS: " + std::to_string( games_per_second ) + " | " + name1
                                           + " Avg Loss: " + std::to_string( game_count > 0 ? total_loss1 / game_count : 0.0 ) + " | " + name2
                                           + " Avg Loss: " + std::to_string( game_count > 0 ? total_loss2 / game_count : 0.0 ) + " | " + name1
                                           + " Avg Reward: " + std::to_string( epoch_total_reward1 / game_count ) + " | " + name2
                                           + " Avg Reward: " + std::to_string( epoch_total_reward2 / game_count ) + " | Games Played: " + std::to_string( game_count );

                std::cout << epochSummary << std::endl;

                {
                    std::ofstream log_file( "training_log.txt", std::ios::app );
                    if ( log_file.is_open() ) {
                        log_file << epochSummary << std::endl;
                    }
                }
                if ( epoch % 10 == 0 ) {
                    NNAI::saveModel( model1, "model_" + name1 + ".pt" );
                    NNAI::saveModel( model2, "model_" + name2 + ".pt" );
                }
            }
        }

        catch ( const fheroes2::InvalidDataResources & ex ) {
            ERROR_LOG( ex.what() )
            displayMissingResourceWindow();
            return EXIT_FAILURE;
        }
    }
    catch ( const std::exception & ex ) {
        ERROR_LOG( "Exception '" << ex.what() << "' occurred during application runtime." )
        return EXIT_FAILURE;
    }
    catch ( ... ) {
        ERROR_LOG( "An unknown exception occurred during application runtime." )
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int default_main( int argc, char ** argv )
{
    fheroes2::cursor();
// SDL2main.lib converts argv to UTF-8, but this application expects ANSI, use the original argv
#if defined( _WIN32 )
    assert( argc == __argc );

    argv = __argv;
#else
    (void)argc;
#endif

    try {
        const fheroes2::HardwareInitializer hardwareInitializer;
        Logging::InitLog();

        COUT( GetCaption() )

        Settings & conf = Settings::Get();
        conf.SetProgramPath( argv[0] );

        InitConfigDir();
        InitDataDir();
        ReadConfigs();

        std::set<fheroes2::SystemInitializationComponent> coreComponents{ fheroes2::SystemInitializationComponent::Audio,
                                                                          fheroes2::SystemInitializationComponent::Video };

#if defined( TARGET_PS_VITA ) || defined( TARGET_NINTENDO_SWITCH )
        coreComponents.emplace( fheroes2::SystemInitializationComponent::GameController );
#endif

        const fheroes2::CoreInitializer coreInitializer( coreComponents );

        DEBUG_LOG( DBG_GAME, DBG_INFO, conf.String() )

        const DisplayInitializer displayInitializer;
        const DataInitializer dataInitializer;

        ListFiles midiSoundFonts;

        midiSoundFonts.Append( Settings::FindFiles( System::concatPath( "files", "soundfonts" ), ".sf2", false ) );
        midiSoundFonts.Append( Settings::FindFiles( System::concatPath( "files", "soundfonts" ), ".sf3", false ) );

#ifdef WITH_DEBUG
        for ( const std::string & file : midiSoundFonts ) {
            DEBUG_LOG( DBG_GAME, DBG_INFO, "MIDI SoundFont to load: " << file )
        }
#endif

        const AudioManager::AudioInitializer audioInitializer( dataInitializer.getOriginalAGGFilePath(), dataInitializer.getExpansionAGGFilePath(), midiSoundFonts );

        // Load palette.
        fheroes2::setGamePalette( AGG::getDataFromAggFile( "KB.PAL" ) );
        fheroes2::Display::instance().changePalette( nullptr, true );

        // init game data
        Game::Init();

        conf.setGameLanguage( conf.getGameLanguage() );

        if ( conf.isShowIntro() ) {
            fheroes2::showTeamInfo();

            Video::ShowVideo( "NWCLOGO.SMK", Video::VideoAction::PLAY_TILL_VIDEO_END );
            Video::ShowVideo( "CYLOGO.SMK", Video::VideoAction::PLAY_TILL_VIDEO_END );
            Video::ShowVideo( "H2XINTRO.SMK", Video::VideoAction::PLAY_TILL_VIDEO_END );
        }

        try {
            const CursorRestorer cursorRestorer( true, Cursor::POINTER );

            Game::mainGameLoop( conf.isFirstGameRun(), isProbablyDemoVersion() );
        }
        catch ( const fheroes2::InvalidDataResources & ex ) {
            ERROR_LOG( ex.what() )
            displayMissingResourceWindow();
            return EXIT_FAILURE;
        }
    }
    catch ( const std::exception & ex ) {
        ERROR_LOG( "Exception '" << ex.what() << "' occurred during application runtime." )
        return EXIT_FAILURE;
    }
    catch ( ... ) {
        ERROR_LOG( "An unknown exception occurred during application runtime." )
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

#include <ostream>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADevice.h>
#include <torch/torch.h>

int main( int argc, char ** argv )
{
    NNAI::device = torch::Device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU );
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << "Device: " << NNAI::device << std::endl;

    NNAI::initializeGlobalModels();
    if ( NNAI::isTraining ) {
        auto model1 = *NNAI::g_model_blue;
        auto model2 = *NNAI::g_model_red;

        model1->to( NNAI::device ); // Ensure model is on device
        model2->to( NNAI::device );

        return NNAI::training_main( argc, argv, /*epochs = */ 2500, 0.0005, NNAI::device, /*games per epoch = */ 200 );
    }

    NNAI::g_model1 = std::make_shared<NNAI::BattleLSTM>( *NNAI::g_model_blue );
    NNAI::g_model2 = std::make_shared<NNAI::BattleLSTM>( *NNAI::g_model_red );
    NNAI::g_model1->get()->to( NNAI::device );
    NNAI::g_model2->get()->to( NNAI::device );

    return default_main( argc, argv );
}
