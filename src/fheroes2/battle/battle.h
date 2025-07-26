/***************************************************************************
 *   fheroes2: https://github.com/ihhub/fheroes2                           *
 *   Copyright (C) 2019 - 2024                                             *
 *                                                                         *
 *   Free Heroes2 Engine: http://sourceforge.net/projects/fheroes2         *
 *   Copyright (C) 2010 by Andrey Afletdinov <fheroes2@gmail.com>          *
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
#ifndef H2BATTLE_H
#define H2BATTLE_H

#include <cstdint>
#include <iostream>
#include <vector>

class Army;
class Funds;
class HeroBase;

namespace Battle
{
    class Unit;

    enum
    {
        RESULT_LOSS = 0x01,
        RESULT_RETREAT = 0x02,
        RESULT_SURRENDER = 0x04,
        RESULT_WINS = 0x80
    };

    struct Result
    {
        uint32_t army1{ 0 };
        uint32_t army2{ 0 };
        uint32_t exp1{ 0 };
        uint32_t exp2{ 0 };
        float AIScore1{ 0 };
        float AIScore2{ 0 };
        float AIScoreBalance{ 0 };
        uint32_t killed{ 0 };

        bool AttackerWins() const;
        bool DefenderWins() const;
        uint32_t AttackerResult() const;
        uint32_t DefenderResult() const;
        uint32_t GetExperienceAttacker() const;
        uint32_t GetExperienceDefender() const;

        // Define the operator<< for Result
        friend std::ostream & operator<<( std::ostream & os, const Result & result )
        {
            os << "  AIScore Balance: " << result.AIScoreBalance;
            return os;

            // os << "Result:" << std::endl;
            // os << "  Army 1: " << result.army1 << std::endl;
            // os << "  Army 2: " << result.army2 << std::endl;
            // os << "  Experience 1: " << result.exp1 << std::endl;
            // os << "  Experience 2: " << result.exp2 << std::endl;
            // os << "  AIScore 1: " << result.AIScore1 << std::endl;
            // os << "  AIScore 2: " << result.AIScore2 << std::endl;
            // os << "  AIScore Balance: " << result.AIScoreBalance << std::endl;
            // os << "  Killed: " << result.killed << std::endl;
            // os << "  Attacker Wins: " << ( result.AttackerWins() ? "Yes" : "No" ) << std::endl;
            // os << "  Defender Wins: " << ( result.DefenderWins() ? "Yes" : "No" ) << std::endl;
            // os << "  Attacker Result: " << result.AttackerResult() << std::endl;
            // os << "  Defender Result: " << result.DefenderResult() << std::endl;
            // os << "  Experience Attacker: " << result.GetExperienceAttacker() << std::endl;
            // os << "  Experience Defender: " << result.GetExperienceDefender() << std::endl;
            // return os;
        }
    };

    // Main entry point
    Result Loader( Army &, Army &, int32_t );

    // 🆕 New sub-functions to support modular Loader:
    bool PrepareBattle( Army & army1, Army & army2, int32_t mapsindex, Result & result, bool & isHumanBattle, uint32_t & initialSpellPoints1,
                        uint32_t & initialSpellPoints2, Funds & initialFunds1, Funds & initialFunds2, HeroBase *& commander1, HeroBase *& commander2 );
    Result ExecuteBattleLoop( Army & army1, Army & army2, int32_t mapsindex, bool showBattle, const uint32_t battleSeed, const Funds & initialFunds1,
                              const Funds & initialFunds2, HeroBase * commander1, HeroBase * commander2, bool isHumanBattle );
    void FinalizeBattleResult( Result & result, Army & army1, Army & army2, HeroBase * commander1, HeroBase * commander2 );

    struct TargetInfo
    {
        Unit * defender = nullptr;
        uint32_t damage = 0;
        uint32_t killed = 0;
        bool resist = false;

        TargetInfo() = default;

        explicit TargetInfo( Unit * defender_ )
            : defender( defender_ )
        {}

        static bool isFinishAnimFrame( const TargetInfo & info );
    };

    struct TargetsInfo : public std::vector<TargetInfo>
    {
        TargetsInfo() = default;
    };

    enum MonsterState : uint32_t
    {
        TR_RESPONDED = 0x00000001,
        TR_MOVED = 0x00000002,
        TR_SKIP = 0x00000004,

        LUCK_GOOD = 0x00000100,
        LUCK_BAD = 0x00000200,
        MORALE_GOOD = 0x00000400,
        MORALE_BAD = 0x00000800,

        CAP_TOWER = 0x00001000,
        CAP_SUMMONELEM = 0x00002000,
        CAP_MIRROROWNER = 0x00004000,
        CAP_MIRRORIMAGE = 0x00008000,

        SP_BLOODLUST = 0x00020000,
        SP_BLESS = 0x00040000,
        SP_HASTE = 0x00080000,
        SP_SHIELD = 0x00100000,
        SP_STONESKIN = 0x00200000,
        SP_DRAGONSLAYER = 0x00400000,
        SP_STEELSKIN = 0x00800000,

        SP_ANTIMAGIC = 0x01000000,

        SP_CURSE = 0x02000000,
        SP_SLOW = 0x04000000,
        SP_BERSERKER = 0x08000000,
        SP_HYPNOTIZE = 0x10000000,
        SP_BLIND = 0x20000000,
        SP_PARALYZE = 0x40000000,
        SP_STONE = 0x80000000,

        IS_GOOD_MAGIC = SP_BLOODLUST | SP_BLESS | SP_HASTE | SP_SHIELD | SP_STONESKIN | SP_DRAGONSLAYER | SP_STEELSKIN,
        IS_BAD_MAGIC = SP_CURSE | SP_SLOW | SP_BERSERKER | SP_HYPNOTIZE | SP_BLIND | SP_PARALYZE | SP_STONE,
        IS_MAGIC = IS_GOOD_MAGIC | IS_BAD_MAGIC | SP_ANTIMAGIC,

        IS_PARALYZE_MAGIC = SP_PARALYZE | SP_STONE,
        IS_MIND_MAGIC = SP_BERSERKER | SP_HYPNOTIZE | SP_BLIND | SP_PARALYZE,
    };

    enum class CastleDefenseStructure : int
    {
        NONE = 0,
        WALL1 = 1,
        WALL2 = 2,
        WALL3 = 3,
        WALL4 = 4,
        TOWER1 = 5,
        TOWER2 = 6,
        BRIDGE = 7,
        CENTRAL_TOWER = 8,
        TOP_BRIDGE_TOWER = 9,
        BOTTOM_BRIDGE_TOWER = 10
    };
}

#endif
