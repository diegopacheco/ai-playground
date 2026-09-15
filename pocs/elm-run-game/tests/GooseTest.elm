module GooseTest exposing (suite)

import Expect
import Game.Config as Config
import Game.Goose as Goose
import Test exposing (Test, describe, test)


fall : Int -> Float -> Goose.Goose -> Goose.Goose
fall frames floor goose =
    List.foldl (\_ g -> Goose.step (1 / 60) floor g) goose (List.range 1 frames)


suite : Test
suite =
    describe "Goose"
        [ test "a jump from the ground launches the goose with full jump power" <|
            \_ ->
                (Goose.jump Goose.init).vy
                    |> Expect.equal Config.jumpVelocity
        , test "the second press in the air is a weaker flap, the double jump" <|
            \_ ->
                (Goose.init |> Goose.jump |> Goose.jump).vy
                    |> Expect.equal Config.flapVelocity
        , test "a third press does nothing so the goose cannot fly forever" <|
            \_ ->
                let
                    flapped =
                        Goose.init |> Goose.jump |> Goose.jump |> fall 5 0
                in
                Goose.jump flapped
                    |> Expect.equal flapped
        , test "releasing jump early cuts the climb so short taps give short hops" <|
            \_ ->
                (Goose.init |> Goose.jump |> Goose.release).vy
                    |> Expect.equal Config.jumpCut
        , test "gravity brings the goose back to the floor and restores its jumps" <|
            \_ ->
                let
                    landed =
                        Goose.init |> Goose.jump |> fall 120 0
                in
                Expect.equal ( landed.y, landed.grounded, landed.jumps ) ( 0, True, 0 )
        , test "walking off a ledge spends the ground jump so only the flap remains" <|
            \_ ->
                let
                    onRoof =
                        { init | y = 24 }

                    init =
                        Goose.init
                in
                (Goose.step (1 / 60) 0 onRoof).jumps
                    |> Expect.equal 1
        , test "ducking only works on the ground" <|
            \_ ->
                (Goose.init |> Goose.jump |> Goose.setDuck True).ducking
                    |> Expect.equal False
        , test "a ducking goose has a shorter hitbox so seagulls fly over it" <|
            \_ ->
                let
                    standing =
                        Goose.box 0 Goose.init

                    ducking =
                        Goose.box 0 (Goose.setDuck True Goose.init)
                in
                Expect.lessThan standing.h ducking.h
        , test "invincibility wears off after the hurt time" <|
            \_ ->
                Goose.init
                    |> Goose.hurt
                    |> fall (ceiling (Config.hurtTime * 60) + 1) 0
                    |> Goose.isInvincible
                    |> Expect.equal False
        ]
