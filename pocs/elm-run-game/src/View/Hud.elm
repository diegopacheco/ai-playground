module View.Hud exposing (view)

import Game.Core as Core exposing (Game, Screen(..))
import Game.District as District
import Svg exposing (Svg)
import Svg.Lazy exposing (lazy)
import View.Pixel exposing (place)
import View.Sprites as Sprites
import View.Text as Text exposing (Anchor(..))


view : Game -> Svg msg
view game =
    Svg.g []
        (if game.screen == Title then
            []

         else
            [ Text.label Start 6 13 7 "#fbf8f0" ("SCORE " ++ Text.padScore (Core.score game))
            , place 6 18 (lazy Sprites.draw Sprites.sourdough)
            , Text.label Start 21 26 6 "#fbf8f0" ("x" ++ String.fromInt game.bread)
            , Text.label End 314 13 7 "#f6c945" ("HI " ++ Text.padScore game.best)
            , hearts game.lives
            , banner game
            ]
        )


hearts : Int -> Svg msg
hearts lives =
    Svg.g []
        (List.range 1 lives
            |> List.map (\i -> place (314 - toFloat (i * 9)) 19 (lazy Sprites.draw Sprites.heart))
        )


banner : Game -> Svg msg
banner game =
    if game.screen == Playing && District.isEntering game.distance then
        Text.label Middle 160 48 8 "#fbf8f0" (District.name (District.at game.distance))

    else
        Svg.g [] []
