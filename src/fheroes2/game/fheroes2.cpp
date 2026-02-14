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
#include <fstream>
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

#include <fstream>

#include "NN_ai.h";
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

/*
Main Training loop for the game.
It chcages default game loop to run training games.
This will instatnly loop through the game and collect data for training n times specified by NNAI::TrainingLoopsCount.
This is used to train the neural network models for the game.
It is not intended to be used by the end user.

Modified from the original fheroes2::Game::mainGameLoop() by
Milan Wr\F3blewski for the purpose of Engineer thesis.
*/

int NNAI::training_main( int argc, char ** argv, int64_t num_series, double learning_rate, torch::Device device, int64_t episodes_per_series )
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
        // ListFiles midiSoundFonts;
        // midiSoundFonts.Append( Settings::FindFiles( System::concatPath( "files", "soundfonts" ), ".sf2", false ) );
        // midiSoundFonts.Append( Settings::FindFiles( System::concatPath( "files", "soundfonts" ), ".sf3", false ) );
#ifdef WITH_DEBUG
        for ( const std::string & file : midiSoundFonts ) {
            DEBUG_LOG( DBG_GAME, DBG_INFO, "MIDI SoundFont to load: " << file )
        }
#endif
        // const AudioManager::AudioInitializer audioInitializer( dataInitializer.getOriginalAGGFilePath(), dataInitializer.getExpansionAGGFilePath(), midiSoundFonts );
        fheroes2::setGamePalette( AGG::getDataFromAggFile( "KB.PAL" ) );
        fheroes2::Display::instance().changePalette( nullptr, true );
        Game::Init();
        conf.setGameLanguage( conf.getGameLanguage() );

        const CursorRestorer cursorRestorer( true, Cursor::POINTER );

        const bool allowTraining = !NNAI::isRunningExperiments;

        double total_elapsed_seconds = 0.0;

        std::stringstream log_buffer; // Buffer to hold log messages

        // Initialize Q-models and per-color replay buffers
        initialize_qmodels( device );

        // Build model entries: include color id so we can pick buffer easily
        struct ModelEntry
        {
            std::shared_ptr<QNetwork> model;
            std::shared_ptr<QNetwork> target;
            std::string name;
            int color; // color id used in game (e.g., 0x01 = blue, 0x04 = red)
        };

        std::vector<ModelEntry> models;

        if ( g_qmodel_blue )
            models.push_back( { g_qmodel_blue, g_target_blue, "blue", 0x01 } );
        if ( g_qmodel_red )
            models.push_back( { g_qmodel_red, g_target_red, "red", 0x04 } );

        if ( models.empty() ) {
            std::cerr << "No Q-models loaded. Aborting training." << std::endl;
            return EXIT_FAILURE;
        }

        // Create one optimizer per model (wrap new Adam into unique_ptr<Optimizer>)
        std::vector<std::pair<std::shared_ptr<QNetwork>, std::unique_ptr<torch::optim::Optimizer>>> optimizers;
        for ( auto & me : models ) {
            if ( allowTraining ) {
                me.model->get()->train();
                auto * adam_ptr = new torch::optim::Adam( me.model->get()->parameters(), torch::optim::AdamOptions( learning_rate ) );
                optimizers.emplace_back( me.model, std::unique_ptr<torch::optim::Optimizer>( adam_ptr ) );
            }
            else {
                me.model->get()->eval(); // inference-only
                for ( auto & p : me.model->get()->parameters() ) {
                    p.requires_grad_( false ); // hard freeze
                }
            }
        }

        double DynamicEPS_Decay = ( EPS_START - EPS_END ) / ( num_series * episodes_per_series );

        std::cout << "DynamicEPS_Decay was calculated to be: " << DynamicEPS_Decay << std::endl << "Start at:" << EPS_START << "End at: " << EPS_END << std::endl;

        // ===== TRAINING LOOP: SERIES / EPISODES =====
        try {
            int blue_start = isRunningExperiments ? 1 : -1;
            int blue_end = isRunningExperiments ? 5 : -1;

            int red_start = isRunningExperiments ? 1 : -1;
            int red_end = isRunningExperiments ? 5 : -1;

            for ( int blue = blue_start; blue <= blue_end; ++blue ) {
                for ( int red = red_start; red <= red_end; ++red ) {
                    blue_monster_count = blue;
                    red_monster_count = red;
                    // ======SERIES========
                    for ( int64_t series = 0; series < num_series; ++series ) {
                        auto series_start = std::chrono::steady_clock::now();

                        float series_total_reward = 0.0f;

                        int blue_wins = 0;
                        int red_wins = 0;

                        // ---- Play episodes in this series ----
                        for ( int64_t ep = 0; ep < episodes_per_series; ++ep ) {
                            // Play one full game (self-play)

                            trainingGameLoop( false, isProbablyDemoVersion() ); // should push to replay buffers

                            double _blueReward = g_replay_buffer_blue->get_last_reward();
                            double _redReward = g_replay_buffer_red->get_last_reward();

                            series_total_reward += _blueReward;
                            series_total_reward += _redReward;

                            if ( _blueReward > _redReward ) {
                                ++blue_wins;
                            }
                            else if ( _blueReward < _redReward ) {
                                ++red_wins;
                            }

                            if ( allowTraining ) {
                                // ---- Update Q after each game ----
                                for ( size_t mi = 0; mi < models.size(); ++mi ) {
                                    auto & me = models[mi];
                                    auto & opt_pair = optimizers[mi];
                                    auto & model_ptr = opt_pair.first;
                                    auto & optimizer_ptr = opt_pair.second;

                                    std::shared_ptr<ReplayBuffer> buf = nullptr;
                                    if ( me.color == 0x01 )
                                        // buf = me.isRanged ? g_replay_buffer_blue_ranged : g_replay_buffer_blue;
                                        buf = g_replay_buffer_blue;
                                    else if ( me.color == 0x04 )
                                        // buf = me.isRanged ? g_replay_buffer_red_ranged : g_replay_buffer_red;
                                        buf = g_replay_buffer_red;

                                    if ( model_ptr && optimizer_ptr && buf ) {
                                        try {
                                            optimize_model( *model_ptr, *optimizer_ptr, buf, GAMMA, device );
                                        }
                                        catch ( const std::exception & ex ) {
                                            std::cerr << "optimize_model exception for " << me.name << ": " << ex.what() << std::endl;
                                        }
                                    }
                                }
                            }
                        }

                        if ( allowTraining ) {
                            // ---- Soft-update after series completes ----
                            for ( auto & me : models ) {
                                if ( me.target && me.model ) {
                                    try {
                                        soft_update_target( *me.model, *me.target, TAU );
                                    }
                                    catch ( ... ) {
                                    }
                                }
                            }
                        }

                        // ---- Logging for this series ----
                        auto series_end = std::chrono::steady_clock::now();
                        std::chrono::duration<double> d = series_end - series_start;
                        int pct = int( ( ( series + 1.0 ) / num_series ) * 100.0 );

                        std::string msg = "Series " + std::to_string( series + 1 ) + "/" + std::to_string( num_series ) + " (" + std::to_string( pct ) + "%)"
                                          + " | Time: " + std::to_string( d.count() ) + "s" + " | Episodes: " + std::to_string( episodes_per_series )
                                          + " | Avg Reward: " + std::to_string( series_total_reward / (double)episodes_per_series )
                                          + " | Blue win percantage: " + ( std::to_string( ( (double)( blue_wins ) / episodesPerSeries ) * 100 ) )
                                          + " | Red win percantage: " + ( std::to_string( ( (double)( red_wins ) / episodesPerSeries ) * 100 ) );
                        if ( isRunningExperiments ) {
                            msg += " | Blue Troops: " + std::to_string( blue_monster_count ) + " | Red Troops: " + std::to_string( red_monster_count );
                            msg += " | Enemy Type: ";
                            switch ( enemyType ) {
                            case -1:
                                msg += "NNAI Enemy";
                                break;
                            case 0:
                                msg += "Default Enemy";
                                break;
                            case 1:
                                msg += "Aggressive Enemy";
                                break;
                            case 2:
                                msg += "Random Enemy";
                                break;
                            case 3:
                                msg += "NNAI Enemy";
                                break;
                            default:
                                msg += "Unknown Enemy Type";
                            }
                        }

                        std::cout << msg << std::endl;
                        log_buffer << msg << std::endl;

                        // ---- Save models/log every series ----
                        std::ofstream log( "training_log.txt", std::ios::app );
                        log << log_buffer.str();
                        log_buffer.str( "" );
                        log.close();

                        if ( allowTraining ) {
                            for ( auto & me : models ) {
                                save_qmodel( *me.model, "qmodel_" + me.name + ".pt" );
                            }

                            // ---- Decay epsilon here ----
                            epsilon = std::max( EPS_END, epsilon - DynamicEPS_Decay );
                        }
                    }
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

#include <torch/torch.h>

int main( int argc, char ** argv )
{
    // Prompt user for training mode at the very beginning
    std::cout << "Enable Neural Network training mode? (y/n): ";
    char train_input = 'n';
    std::cin >> train_input;
    // Prompt user for debug logs skiping
    std::cout << "Skip Debug log? (y/n): ";
    char skip_debug_input = 'n';
    std::cin >> skip_debug_input;

    std::cout << "NN Enemy? (y/n): ";
    char auto_nn_enemy_input = 'n';
    std::cin >> auto_nn_enemy_input;

    std::cout << "Disable experiment mode? (y/n): ";
    char disable_experiment_input = 'n';
    std::cin >> disable_experiment_input;

    char enemy_choice_input = 'n';

    if ( auto_nn_enemy_input == 'y' || auto_nn_enemy_input == 'Y' ) {
        NNAI::enemyType = -1; // Neural AI Enemy
    }
    else {
        std::cout << "Select Enemy: " << std::endl;
        std::cout << "0 -> Default Enemy" << std::endl;
        std::cout << "1 -> Agressive Enemy" << std::endl;
        std::cout << "2 -> Random Enemy" << std::endl;
        std::cout << "3 -> NNAI Enemy" << std::endl;
        while ( !( enemy_choice_input == '0' || enemy_choice_input == '1' || enemy_choice_input == '2' || enemy_choice_input == '3' ) ) {
            std::cout << "ENEMY: ";
            std::cin >> enemy_choice_input;
        }
        NNAI::enemyType = static_cast<int>( enemy_choice_input - '0' );
    }

    // Set isTraining based on user input
    // Note: isTraining must be non-const and not constexpr in NN_ai.h for this to work!
    NNAI::isTraining = ( train_input == 'y' || train_input == 'Y' );
    NNAI::skipDebugLog = ( skip_debug_input == 'y' || skip_debug_input == 'Y' );
    NNAI::isRunningExperiments = !( disable_experiment_input == 'y' || disable_experiment_input == 'Y' );

    NNAI::device = torch::Device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU );

    NNAI::device = torch::kCPU; // Force CPU for now, as CUDA is slower in this environment

    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << "Device: " << NNAI::device << std::endl;

    NNAI::episodesPerSeries = 100;
    if ( NNAI::isRunningExperiments ) {
        NNAI::episodesPerSeries = 500;
    }

    if ( NNAI::isTraining ) {
        AI::BattlePlanner::MAX_TURNS_WITHOUT_DEATHS = 5; // Set the max turns without deaths for the planner
        int numSeries = 100;
        if ( NNAI::isRunningExperiments ) {
            numSeries = 1;
        }
        return NNAI::training_main( argc, argv, /*series = */ numSeries, 0.0005, NNAI::device, /*episodes per series = */ NNAI::episodesPerSeries );
    }

    // Initialize Q-models and per-color replay buffers
    NNAI::initialize_qmodels( NNAI::device );
    return default_main( argc, argv );
}
