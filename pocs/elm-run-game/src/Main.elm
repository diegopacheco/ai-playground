module Main exposing (main)

import Browser
import Browser.Events
import Game.Core as Core exposing (Game, Screen(..))
import Game.Input as Input exposing (Action)
import Json.Decode as Decode
import View.Scene as Scene


type Msg
    = Frame Float
    | Act Action


main : Program Int Game Msg
main =
    Browser.element
        { init = \seed -> ( Core.init seed, Cmd.none )
        , update = update
        , view = Scene.view Act
        , subscriptions = subscriptions
        }


update : Msg -> Game -> ( Game, Cmd Msg )
update msg game =
    case msg of
        Frame delta ->
            ( Core.step (delta / 1000) game, Cmd.none )

        Act action ->
            ( Core.apply action game, Cmd.none )


subscriptions : Game -> Sub Msg
subscriptions game =
    Sub.batch
        [ Browser.Events.onKeyDown (Decode.map Act (Input.decoder Input.keyDown))
        , Browser.Events.onKeyUp (Decode.map Act (Input.decoder Input.keyUp))
        , if game.screen == Paused then
            Sub.none

          else
            Browser.Events.onAnimationFrameDelta Frame
        ]
