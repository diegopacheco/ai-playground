module Game.Entity exposing (Entity, Kind(..), box, move, size)

import Game.Collision exposing (Box)
import Game.Config as Config


type Kind
    = Hydrant
    | Cone
    | CableCar
    | Seagull
    | Sourdough


type alias Entity =
    { kind : Kind
    , x : Float
    , y : Float
    }


size : Kind -> { w : Float, h : Float }
size kind =
    case kind of
        Hydrant ->
            { w = 10, h = 14 }

        Cone ->
            { w = 10, h = 11 }

        CableCar ->
            { w = 40, h = 24 }

        Seagull ->
            { w = 16, h = 8 }

        Sourdough ->
            { w = 12, h = 8 }


box : Entity -> Box
box entity =
    let
        s =
            size entity.kind
    in
    { x = entity.x, y = entity.y, w = s.w, h = s.h }


move : Float -> Entity -> Entity
move dt entity =
    case entity.kind of
        Seagull ->
            { entity | x = entity.x - Config.seagullSpeed * dt }

        _ ->
            entity
