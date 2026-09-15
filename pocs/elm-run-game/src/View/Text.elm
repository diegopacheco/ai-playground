module View.Text exposing (Anchor(..), label, padScore)

import Svg exposing (Svg)
import Svg.Attributes as A


type Anchor
    = Start
    | Middle
    | End


label : Anchor -> Float -> Float -> Float -> String -> String -> Svg msg
label anchor x y size color content =
    Svg.g []
        [ textAt anchor (x + 1) (y + 1) size "#1a1423" content
        , textAt anchor x y size color content
        ]


textAt : Anchor -> Float -> Float -> Float -> String -> String -> Svg msg
textAt anchor x y size color content =
    Svg.text_
        [ A.x (String.fromFloat x)
        , A.y (String.fromFloat y)
        , A.fontSize (String.fromFloat size)
        , A.fill color
        , A.fontFamily "'Press Start 2P', monospace"
        , A.textAnchor (anchorName anchor)
        ]
        [ Svg.text content ]


anchorName : Anchor -> String
anchorName anchor =
    case anchor of
        Start ->
            "start"

        Middle ->
            "middle"

        End ->
            "end"


padScore : Int -> String
padScore value =
    String.padLeft 6 '0' (String.fromInt value)
