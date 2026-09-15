module View.Overlay exposing (view)

import Game.Config as Config
import Game.Core as Core exposing (Game, Screen(..))
import Svg exposing (Svg)
import Svg.Attributes as A
import View.Pixel exposing (block)
import View.Text as Text exposing (Anchor(..))


view : Game -> Svg msg
view game =
    case game.screen of
        Title ->
            Svg.g []
                [ Text.label Middle 160 58 20 "#f6c945" "GOOZE RUN"
                , Text.label Middle 160 76 7 "#fbf8f0" "SAN FRANCISCO"
                , blink game.clock (Text.label Middle 160 104 7 "#fbf8f0" "PRESS SPACE TO RUN")
                , Text.label Middle 160 122 5 "#fbf8f0" "SPACE JUMP - AGAIN IN AIR TO FLAP"
                , Text.label Middle 160 132 5 "#fbf8f0" "DOWN DUCK - P PAUSE"
                ]

        Paused ->
            Svg.g []
                [ dim
                , Text.label Middle 160 84 14 "#f6c945" "PAUSED"
                , Text.label Middle 160 104 6 "#fbf8f0" "PRESS P TO RESUME"
                ]

        GameOver ->
            Svg.g []
                [ dim
                , Text.label Middle 160 64 16 "#d7263d" "GAME OVER"
                , Text.label Middle 160 86 7 "#fbf8f0" ("SCORE " ++ Text.padScore (Core.score game))
                , Text.label Middle 160 100 7 "#f6c945" ("BEST  " ++ Text.padScore game.best)
                , if game.clock >= Config.restartDelay then
                    blink game.clock (Text.label Middle 160 124 6 "#fbf8f0" "PRESS ENTER TO RUN AGAIN")

                  else
                    Svg.g [] []
                ]

        Playing ->
            Svg.g [] []


dim : Svg msg
dim =
    Svg.g [ A.opacity "0.55" ] [ block 0 0 Config.width Config.height "#1a1423" ]


blink : Float -> Svg msg -> Svg msg
blink clock content =
    if modBy 2 (floor (clock * 2)) == 0 then
        content

    else
        Svg.g [] []
