module Game.Collision exposing (Box, overlaps, overlapsX, top)


type alias Box =
    { x : Float
    , y : Float
    , w : Float
    , h : Float
    }


overlapsX : Box -> Box -> Bool
overlapsX a b =
    a.x < b.x + b.w && b.x < a.x + a.w


overlaps : Box -> Box -> Bool
overlaps a b =
    overlapsX a b && a.y < top b && b.y < top a


top : Box -> Float
top box =
    box.y + box.h
