module Game.District exposing (District(..), at, isEntering, name)

import Game.Config as Config


type District
    = Mission
    | Chinatown
    | GoldenGate


at : Float -> District
at distance =
    case modBy 3 (floor (distance / Config.districtLength)) of
        0 ->
            Mission

        1 ->
            Chinatown

        _ ->
            GoldenGate


isEntering : Float -> Bool
isEntering distance =
    distance - Config.districtLength * toFloat (floor (distance / Config.districtLength)) < 400


name : District -> String
name district =
    case district of
        Mission ->
            "MISSION DISTRICT"

        Chinatown ->
            "CHINATOWN"

        GoldenGate ->
            "GOLDEN GATE"
