module CollisionTest exposing (suite)

import Expect
import Game.Collision as Collision
import Test exposing (Test, describe, test)


suite : Test
suite =
    describe "Collision"
        [ test "boxes sharing space overlap" <|
            \_ ->
                Collision.overlaps { x = 0, y = 0, w = 10, h = 10 } { x = 5, y = 5, w = 10, h = 10 }
                    |> Expect.equal True
        , test "a goose standing exactly on a roof does not collide with it" <|
            \_ ->
                Collision.overlaps { x = 0, y = 24, w = 11, h = 14 } { x = 0, y = 0, w = 40, h = 24 }
                    |> Expect.equal False
        , test "boxes side by side do not overlap" <|
            \_ ->
                Collision.overlaps { x = 0, y = 0, w = 10, h = 10 } { x = 10, y = 0, w = 10, h = 10 }
                    |> Expect.equal False
        ]
