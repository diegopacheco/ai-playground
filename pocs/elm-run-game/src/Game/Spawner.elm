module Game.Spawner exposing (generate)

import Game.Config as Config
import Game.Entity exposing (Entity, Kind(..))
import Random exposing (Generator)


type alias Pattern =
    Float -> ( List Entity, Float )


generate : Float -> Float -> Random.Seed -> ( List Entity, Float, Random.Seed )
generate x speed seed =
    let
        ( ( pattern, gap ), next ) =
            Random.step (Random.pair patterns (Random.float 90 170)) seed

        ( entities, span ) =
            pattern x
    in
    ( entities, x + span + gap * (0.6 + 0.4 * speed / Config.startSpeed), next )


patterns : Generator Pattern
patterns =
    Random.weighted
        ( 20, single Hydrant )
        [ ( 10, single Cone )
        , ( 15, cones )
        , ( 15, seagull )
        , ( 15, cableCar )
        , ( 15, breadArc )
        ]


single : Kind -> Pattern
single kind x =
    ( [ Entity kind x 0 ], 12 )


cones : Pattern
cones x =
    ( [ Entity Cone x 0, Entity Cone (x + 12) 0 ], 24 )


seagull : Pattern
seagull x =
    ( [ Entity Seagull x 11 ], 16 )


cableCar : Pattern
cableCar x =
    ( [ Entity CableCar x 0, Entity Sourdough (x + 14) 32 ], 40 )


breadArc : Pattern
breadArc x =
    ( [ Entity Sourdough x 20, Entity Sourdough (x + 18) 40, Entity Sourdough (x + 36) 20 ], 48 )
