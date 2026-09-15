module View.Sprites exposing
    ( all
    , cableCar
    , cone
    , draw
    , gooseDuck
    , gooseJump
    , gooseRunA
    , gooseRunB
    , heart
    , hydrant
    , seagullDown
    , seagullUp
    , sourdough
    )

import Svg exposing (Svg)
import View.Pixel as Pixel


draw : List String -> Svg msg
draw rows =
    Pixel.sprite palette rows


palette : Char -> Maybe String
palette char =
    case char of
        'k' ->
            Just "#1a1423"

        'w' ->
            Just "#fbf8f0"

        'g' ->
            Just "#a7a9b4"

        'o' ->
            Just "#f7931e"

        'r' ->
            Just "#d7263d"

        'm' ->
            Just "#7a1f2b"

        't' ->
            Just "#e8b86d"

        'b' ->
            Just "#8a4b22"

        'u' ->
            Just "#2d6cdf"

        'l' ->
            Just "#9fd3f5"

        'y' ->
            Just "#f6c945"

        _ ->
            Nothing


all : List (List String)
all =
    [ gooseRunA, gooseRunB, gooseJump, gooseDuck, hydrant, cone, cableCar, seagullUp, seagullDown, sourdough, heart ]


gooseBody : List String
gooseBody =
    [ "..........kkkk.."
    , ".........kwwwwk."
    , ".........kwwkwk."
    , ".........kwwwwoo"
    , "..........kwwwoo"
    , "..........kwwk.."
    , "..........kwwk.."
    ]


gooseRunA : List String
gooseRunA =
    gooseBody
        ++ [ "..k.......kwwk.."
           , ".kwk....kkkwwk.."
           , ".kwwkkkkwwwwwk.."
           , "..kwwwwwwwgwwk.."
           , "..kwwwgggwwwk..."
           , "...kwwwwwwwk...."
           , "....kkkkkkk....."
           , ".....o..o......."
           , ".....oo.oo......"
           ]


gooseRunB : List String
gooseRunB =
    gooseBody
        ++ [ "..k.......kwwk.."
           , ".kwk....kkkwwk.."
           , ".kwwkkkkwwwwwk.."
           , "..kwwwwwwwgwwk.."
           , "..kwwwgggwwwk..."
           , "...kwwwwwwwk...."
           , "....kkkkkkk....."
           , "....o....o......"
           , "....oo...oo....."
           ]


gooseJump : List String
gooseJump =
    gooseBody
        ++ [ "..k...kk..kwwk.."
           , ".kwk.kggkkkwwk.."
           , ".kwwkkkkwwwwwk.."
           , "..kwwwwwwwgwwk.."
           , "..kwwwgggwwwk..."
           , "...kwwwwwwwk...."
           , "....kkkkkkk....."
           , "......oo........"
           , "................"
           ]


gooseDuck : List String
gooseDuck =
    [ "..........kkkk.."
    , ".k......kkwwwwk."
    , "kwkkkkkkwwwwkwk."
    , "kwwwwwwwwwwwwwoo"
    , ".kwwwwgggwwwwwoo"
    , ".kwwwwwwwwwwkk.."
    , "..kwwwwwwwwk...."
    , "...kkkkkkkk....."
    , "....o...o......."
    , "...oo..oo......."
    ]


hydrant : List String
hydrant =
    [ "...kkkk..."
    , "..kuuuuk.."
    , "..kuuuuk.."
    , ".kkkkkkkk."
    , "kwwwwwwwgk"
    , "kwwwwwwwgk"
    , ".kwwwwwgk."
    , ".kwwkwwgk."
    , ".kwwwwwgk."
    , ".kwwwwwgk."
    , ".kwwwwwgk."
    , ".kwwwwwgk."
    , "kkkkkkkkkk"
    , "kggggggggk"
    ]


cone : List String
cone =
    [ "....kk...."
    , "...kook..."
    , "...kook..."
    , "..kowwok.."
    , "..kowwok.."
    , "..kooook.."
    , ".kowwwwok."
    , ".kooooook."
    , "kooooooook"
    , "kkkkkkkkkk"
    , "kggggggggk"
    ]


cableCar : List String
cableCar =
    let
        edge fill =
            "k" ++ String.repeat 38 fill ++ "k"

        wheels =
            "....kggk" ++ String.repeat 24 "." ++ "kggk...."
    in
    [ "." ++ String.repeat 38 "k" ++ "."
    , edge "m"
    , edge "m"
    , edge "t"
    ]
        ++ List.repeat 8 ("kt" ++ String.repeat 6 "lllltt" ++ "tk")
        ++ [ edge "t"
           , edge "m"
           , edge "r"
           , edge "y"
           , edge "r"
           , edge "r"
           , edge "m"
           , "." ++ String.repeat 38 "k" ++ "."
           , wheels
           , "...kggggk" ++ String.repeat 22 "." ++ "kggggk..."
           , wheels
           , "....kkkk" ++ String.repeat 24 "." ++ "kkkk...."
           ]


seagullUp : List String
seagullUp =
    [ "kk............kk"
    , ".kgk........kgk."
    , "..kwk..kk..kwk.."
    , "...kwkkwwkkwk..."
    , ".kkwwwwwwwwwwk.."
    , "ookwkwwwwwwwwk.."
    , "..kkwwwwwwwwkkk."
    , "....kkkkkkkk...."
    ]


seagullDown : List String
seagullDown =
    [ "................"
    , "................"
    , ".......kk......."
    , "......kwwk......"
    , ".kkkkwwwwwwkkkk."
    , "ookwkwwwwwwwwk.."
    , ".kgkkwwwwwwkkgk."
    , "kgk.kkkkkkkk.kgk"
    ]


sourdough : List String
sourdough =
    [ "...kkkkkk..."
    , "..kttbbttk.."
    , ".kttttttttk."
    , "kttbttttbttk"
    , "kttttttttttk"
    , "kbttttttttbk"
    , ".kbbbbbbbbk."
    , "..kkkkkkkk.."
    ]


heart : List String
heart =
    [ ".kk.kk."
    , "krrkrrk"
    , "krrrrrk"
    , ".krrrk."
    , "..krk.."
    , "...k..."
    ]
