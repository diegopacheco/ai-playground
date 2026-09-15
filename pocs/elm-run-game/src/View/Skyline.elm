module View.Skyline exposing (view)

import Game.Config as Config
import Game.District exposing (District(..))
import Svg exposing (Svg)
import Svg.Attributes as A
import Svg.Lazy exposing (lazy)
import View.Pixel exposing (block)


view : District -> Float -> Svg msg
view district scroll =
    Svg.g []
        [ lazy sky district
        , layer 0.2 scroll (lazy far district)
        , layer 0.5 scroll (lazy mid district)
        , layer 1 scroll (lazy ground district)
        ]


layer : Float -> Float -> Svg msg -> Svg msg
layer factor scroll content =
    let
        travel =
            scroll * factor

        offset =
            travel - Config.width * toFloat (floor (travel / Config.width))
    in
    Svg.g [ A.transform ("translate(" ++ String.fromInt -(round offset) ++ " 0)") ]
        [ content
        , Svg.g [ A.transform "translate(320 0)" ] [ content ]
        ]


sky : District -> Svg msg
sky district =
    let
        colors =
            case district of
                Mission ->
                    [ "#ff7b54", "#ff9e6d", "#ffb88a", "#ffd29d" ]

                Chinatown ->
                    [ "#2b1d4a", "#43306b", "#6b4a8a", "#a8658f" ]

                GoldenGate ->
                    [ "#5fb4f0", "#7fc6f5", "#a4d8f8", "#cdeafb" ]
    in
    Svg.g [] (List.indexedMap (\i color -> block 0 (toFloat i * 38) Config.width 38 color) colors)


far : District -> Svg msg
far district =
    case district of
        Mission ->
            Svg.g [] [ disc 80 70 14 "#ffe08a", sutroTower 196, hills "#6d8f5a" ]

        Chinatown ->
            Svg.g [] (disc 60 40 9 "#f4f1d0" :: stars ++ [ hills "#3a2d52", pyramid 150, coitTower 250 ])

        GoldenGate ->
            Svg.g [] [ hills "#4f7d5b", block 0 118 Config.width 32 "#2f6f9f", bridge ]


hills : String -> Svg msg
hills color =
    Svg.g []
        (List.range 0 79
            |> List.map
                (\i ->
                    let
                        x =
                            toFloat (i * 4)

                        h =
                            34 + 10 * sin (turns (x / 160)) + 6 * sin (turns (x / 64))
                    in
                    block x (150 - toFloat (round h)) 4 (toFloat (round h)) color
                )
        )


disc : Float -> Float -> Int -> String -> Svg msg
disc cx cy radius color =
    Svg.g []
        (List.range -radius radius
            |> List.map
                (\dy ->
                    let
                        half =
                            toFloat (round (sqrt (toFloat (radius * radius - dy * dy))))
                    in
                    block (cx - half) (cy + toFloat dy) (half * 2) 1 color
                )
        )


stars : List (Svg msg)
stars =
    [ ( 20, 12 ), ( 110, 20 ), ( 180, 8 ), ( 230, 30 ), ( 290, 14 ), ( 140, 44 ), ( 260, 56 ) ]
        |> List.map (\( x, y ) -> block x y 1 1 "#fff6d5")


sutroTower : Float -> Svg msg
sutroTower x =
    Svg.g []
        (List.concatMap (\leg -> List.map (towerSegment (x + leg)) (List.range 0 8)) [ 0, 10, 20 ]
            ++ [ block (x - 2) 70 26 2 "#3b3b45", block (x - 2) 90 26 2 "#3b3b45" ]
        )


towerSegment : Float -> Int -> Svg msg
towerSegment x i =
    block x (60 + toFloat (i * 6)) 2 6
        (if modBy 2 i == 0 then
            "#d64545"

         else
            "#f4f4f4"
        )


pyramid : Float -> Svg msg
pyramid cx =
    Svg.g []
        (block (cx - 1) 30 2 10 "#d9d4c7"
            :: (List.range 0 38
                    |> List.map
                        (\i ->
                            let
                                half =
                                    1 + toFloat (round (toFloat i * 0.3))
                            in
                            block (cx - half) (40 + toFloat (i * 2)) (half * 2) 2 "#d9d4c7"
                        )
               )
        )


coitTower : Float -> Svg msg
coitTower x =
    Svg.g [] [ block x 84 8 30 "#e8e2d0", block (x - 1) 82 10 3 "#c9c2ad" ]


bridge : Svg msg
bridge =
    Svg.g []
        (List.map cablePixel (List.range 0 159)
            ++ List.map suspender (List.range 0 39)
            ++ [ block 70 50 8 75 "#c0362c"
               , block 230 50 8 75 "#c0362c"
               , block 0 106 Config.width 4 "#9e2b23"
               ]
        )


cableY : Float -> Float
cableY x =
    if x < 74 then
        54 + 46 * ((74 - x) / 74)

    else if x > 234 then
        54 + 46 * ((x - 234) / 86)

    else
        54 + 46 * (1 - ((x - 154) / 80) ^ 2)


cablePixel : Int -> Svg msg
cablePixel i =
    let
        x =
            toFloat (i * 2)
    in
    block x (toFloat (round (cableY x))) 2 2 "#c0362c"


suspender : Int -> Svg msg
suspender i =
    let
        x =
            toFloat (i * 8)

        y =
            toFloat (round (cableY x))
    in
    block x y 1 (106 - y) "#d2584d"


mid : District -> Svg msg
mid district =
    case district of
        Mission ->
            Svg.g [] (List.indexedMap paintedLady [ "#f7b2bd", "#a0d8c5", "#f9e79f", "#b5c7f2", "#f5c396" ])

        Chinatown ->
            Svg.g [] (List.indexedMap shop [ "#b8322e", "#8f2a2a", "#c9503a", "#a13a3a" ] ++ lanterns)

        GoldenGate ->
            Svg.g [] (List.map pine [ 20, 90, 170, 260 ] ++ List.map fog [ 0, 120, 220 ])


paintedLady : Int -> String -> Svg msg
paintedLady i color =
    let
        x =
            toFloat (i * 64 + 10)
    in
    Svg.g []
        (List.map (\r -> block (x + 22 - (2 + toFloat r * 2)) (84 + toFloat (r * 2)) (4 + toFloat r * 4) 2 "#6b4f6b") (List.range 0 9)
            ++ [ block x 104 44 46 color
               , block x 104 44 2 "#fbf8f0"
               , block (x + 7) 111 10 14 "#fbf8f0"
               , block (x + 8) 112 8 12 "#5a6b8c"
               , block (x + 27) 111 10 14 "#fbf8f0"
               , block (x + 28) 112 8 12 "#5a6b8c"
               , block (x + 17) 132 10 18 "#7a4b3a"
               ]
        )


shop : Int -> String -> Svg msg
shop i color =
    let
        x =
            toFloat (i * 80)
    in
    Svg.g []
        ([ block (x + 6) 96 60 54 color
         , block x 90 72 6 "#2e7d5b"
         , block (x - 2) 87 4 4 "#2e7d5b"
         , block (x + 70) 87 4 4 "#2e7d5b"
         ]
            ++ List.concatMap (\col -> List.map (\r -> block (x + 14 + toFloat (col * 18)) (104 + toFloat (r * 18)) 8 10 "#ffcf5a") [ 0, 1 ]) [ 0, 1, 2 ]
        )


lanterns : List (Svg msg)
lanterns =
    block 0 80 Config.width 1 "#3a2a2a"
        :: List.concatMap (\i -> [ block (toFloat (i * 20 + 8)) 81 4 5 "#e0302a", block (toFloat (i * 20 + 9)) 83 2 1 "#ffcf5a" ]) (List.range 0 15)


pine : Float -> Svg msg
pine x =
    Svg.g []
        (block (x + 9) 138 4 12 "#5b3a24"
            :: List.map (\r -> block (x + 11 - toFloat r) (110 + toFloat (r * 2)) (toFloat (r * 2 + 1)) 2 "#2f5d3a") (List.range 0 13)
        )


fog : Float -> Svg msg
fog x =
    Svg.g [ A.opacity "0.85" ]
        [ block x 124 60 8 "#f2f4f7"
        , block (x + 10) 118 36 6 "#f2f4f7"
        , block (x + 20) 114 16 4 "#f2f4f7"
        ]


ground : District -> Svg msg
ground district =
    let
        sidewalk =
            case district of
                Mission ->
                    "#bfb6a8"

                Chinatown ->
                    "#9c8f86"

                GoldenGate ->
                    "#b9b3a6"
    in
    Svg.g []
        ([ block 0 150 Config.width 3 "#d9d4cc"
         , block 0 153 Config.width 12 sidewalk
         , block 0 165 Config.width 15 "#3d3a40"
         , block 0 168 Config.width 1 "#6c6870"
         , block 0 177 Config.width 1 "#6c6870"
         ]
            ++ List.map (\i -> block (toFloat (i * 16)) 153 1 12 "#8e877c") (List.range 0 19)
            ++ List.map (\i -> block (toFloat (i * 40 + 10)) 172 20 2 "#f6c945") (List.range 0 7)
        )
