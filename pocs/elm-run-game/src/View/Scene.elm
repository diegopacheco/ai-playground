module View.Scene exposing (view)

import Game.Config as Config
import Game.Core exposing (Game, Screen(..))
import Game.District as District
import Game.Entity as Entity exposing (Entity, Kind(..))
import Game.Goose as Goose
import Game.Input exposing (Action(..))
import Html exposing (Html)
import Html.Attributes
import Json.Decode as Decode
import Svg exposing (Svg)
import Svg.Attributes as A
import Svg.Events
import Svg.Lazy exposing (lazy)
import View.Hud as Hud
import View.Overlay as Overlay
import View.Pixel exposing (place)
import View.Sprites as Sprites
import View.Skyline as Skyline


view : (Action -> msg) -> Game -> Html msg
view toMsg game =
    Html.div [ Html.Attributes.class "stage" ]
        [ Svg.svg
            [ A.viewBox "0 0 320 180"
            , A.shapeRendering "crispEdges"
            , A.class "screen"
            , Svg.Events.on "pointerdown" (Decode.succeed (toMsg Jump))
            , Svg.Events.on "pointerup" (Decode.succeed (toMsg JumpReleased))
            ]
            [ Skyline.view (District.at game.distance) (scroll game)
            , Svg.g [] (List.map (entity game) game.entities)
            , goose game
            , Hud.view game
            , Overlay.view game
            ]
        ]


scroll : Game -> Float
scroll game =
    if game.screen == Title then
        game.clock * 40

    else
        game.distance


entity : Game -> Entity -> Svg msg
entity game item =
    let
        size =
            Entity.size item.kind
    in
    place (item.x - game.distance) (Config.groundY - item.y - size.h) (lazy Sprites.draw (entitySprite game.clock item.kind))


entitySprite : Float -> Kind -> List String
entitySprite clock kind =
    case kind of
        Hydrant ->
            Sprites.hydrant

        Cone ->
            Sprites.cone

        CableCar ->
            Sprites.cableCar

        Seagull ->
            if frame 6 clock == 0 then
                Sprites.seagullUp

            else
                Sprites.seagullDown

        Sourdough ->
            Sprites.sourdough


goose : Game -> Svg msg
goose game =
    let
        rows =
            gooseSprite game
    in
    if Goose.isInvincible game.goose && frame 10 game.goose.hurtTimer == 0 then
        Svg.g [] []

    else
        place Config.gooseScreenX (Config.groundY - game.goose.y - toFloat (List.length rows)) (lazy Sprites.draw rows)


gooseSprite : Game -> List String
gooseSprite game =
    let
        g =
            game.goose
    in
    if game.screen == Title then
        runFrame (game.clock * 0.6)

    else if not g.grounded then
        Sprites.gooseJump

    else if g.ducking then
        Sprites.gooseDuck

    else
        runFrame (g.runClock * game.speed / Config.startSpeed)


runFrame : Float -> List String
runFrame clock =
    if frame 10 clock == 0 then
        Sprites.gooseRunA

    else
        Sprites.gooseRunB


frame : Float -> Float -> Int
frame rate clock =
    modBy 2 (floor (clock * rate))
