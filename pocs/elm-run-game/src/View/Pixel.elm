module View.Pixel exposing (Palette, block, place, sprite)

import Svg exposing (Svg)
import Svg.Attributes as A


type alias Palette =
    Char -> Maybe String


type alias Run =
    { char : Char
    , start : Int
    , len : Int
    }


sprite : Palette -> List String -> Svg msg
sprite palette rows =
    Svg.g [] (List.concat (List.indexedMap (row palette) rows))


row : Palette -> Int -> String -> List (Svg msg)
row palette y line =
    String.toList line
        |> List.indexedMap Tuple.pair
        |> List.foldl collectRun []
        |> List.filterMap (\run -> Maybe.map (block (toFloat run.start) (toFloat y) (toFloat run.len) 1) (palette run.char))


collectRun : ( Int, Char ) -> List Run -> List Run
collectRun ( index, char ) runs =
    case runs of
        last :: rest ->
            if last.char == char then
                { last | len = last.len + 1 } :: rest

            else
                Run char index 1 :: runs

        [] ->
            [ Run char index 1 ]


block : Float -> Float -> Float -> Float -> String -> Svg msg
block x y w h color =
    Svg.rect
        [ A.x (String.fromFloat x)
        , A.y (String.fromFloat y)
        , A.width (String.fromFloat w)
        , A.height (String.fromFloat h)
        , A.fill color
        ]
        []


place : Float -> Float -> Svg msg -> Svg msg
place x y content =
    Svg.g
        [ A.transform ("translate(" ++ String.fromInt (round x) ++ " " ++ String.fromInt (round y) ++ ")") ]
        [ content ]
