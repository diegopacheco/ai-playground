module CoreTest exposing (suite)

import Expect
import Game.Config as Config
import Game.Core as Core exposing (Game, Screen(..))
import Game.Entity exposing (Entity, Kind(..))
import Game.Input exposing (Action(..))
import Test exposing (Test, describe, test)


playing : Game
playing =
    Core.start (Core.init 42)


withEntity : Kind -> Float -> Game -> Game
withEntity kind y game =
    { game | entities = [ Entity kind (Core.gooseX game) y ], nextSpawn = 1.0e9 }


frame : Game -> Game
frame =
    Core.step (1 / 60)


suite : Test
suite =
    describe "Core"
        [ test "the title screen starts a run on jump" <|
            \_ ->
                (Core.apply Jump (Core.init 1)).screen
                    |> Expect.equal Playing
        , test "running into a hydrant costs one life" <|
            \_ ->
                (playing |> withEntity Hydrant 0 |> frame).lives
                    |> Expect.equal (Config.startLives - 1)
        , test "after a hit the goose is invincible so one obstacle cannot drain every life" <|
            \_ ->
                (playing |> withEntity Hydrant 0 |> frame |> frame |> frame).lives
                    |> Expect.equal (Config.startLives - 1)
        , test "losing the last life ends the run and records the best score" <|
            \_ ->
                let
                    over =
                        { playing | lives = 1, bread = 3 } |> withEntity Cone 0 |> frame
                in
                Expect.equal ( over.screen, over.best > 0 ) ( GameOver, True )
        , test "game over ignores an instant restart so a held jump does not skip the score" <|
            \_ ->
                let
                    over =
                        { playing | lives = 1 } |> withEntity Cone 0 |> frame
                in
                (Core.apply Jump over).screen
                    |> Expect.equal GameOver
        , test "a new run keeps the best score" <|
            \_ ->
                let
                    over =
                        { playing | lives = 1, bread = 3 } |> withEntity Cone 0 |> frame

                    later =
                        { over | clock = Config.restartDelay }
                in
                (Core.apply Confirm later).best
                    |> Expect.equal over.best
        , test "touching sourdough collects it and adds to the score" <|
            \_ ->
                let
                    after =
                        playing |> withEntity Sourdough 0 |> frame
                in
                Expect.equal ( after.bread, List.length after.entities ) ( 1, 0 )
        , test "every 25 sourdough grants an extra life" <|
            \_ ->
                ({ playing | bread = Config.breadPerLife - 1 } |> withEntity Sourdough 0 |> frame).lives
                    |> Expect.equal (Config.startLives + 1)
        , test "a ducking goose slips under a seagull" <|
            \_ ->
                let
                    ducking =
                        Core.apply (Duck True) playing |> frame
                in
                (ducking |> withEntity Seagull 11 |> frame).lives
                    |> Expect.equal Config.startLives
        , test "landing on a seagull from above stomps it and bounces the goose" <|
            \_ ->
                let
                    goose =
                        playing.goose

                    falling =
                        { playing | goose = { goose | y = 17, vy = -100, grounded = False, jumps = 1 } }
                            |> withEntity Seagull 11
                            |> frame
                in
                Expect.equal ( falling.stomps, falling.lives, falling.goose.vy > 0 ) ( 1, Config.startLives, True )
        , test "a cable car roof is a platform the goose can stand on" <|
            \_ ->
                let
                    goose =
                        playing.goose

                    onRoof =
                        { playing | goose = { goose | y = 24, grounded = True } }
                            |> withEntity CableCar 0
                            |> frame
                            |> frame
                in
                Expect.equal ( onRoof.goose.y, onRoof.lives ) ( 24, Config.startLives )
        , test "running into the side of a cable car hurts" <|
            \_ ->
                (playing |> withEntity CableCar 0 |> frame).lives
                    |> Expect.equal (Config.startLives - 1)
        , test "pause freezes the world" <|
            \_ ->
                let
                    paused =
                        Core.apply Pause playing
                in
                frame paused
                    |> Expect.equal paused
        , test "the run speeds up over time but never past the cap" <|
            \_ ->
                let
                    long =
                        List.foldl (\_ g -> Core.step 0.05 { g | lives = 99, goose = Core.start g |> .goose }) playing (List.range 1 4000)
                in
                Expect.equal long.speed Config.maxSpeed
        , test "obstacles are spawned ahead of the goose, never on top of it" <|
            \_ ->
                (frame playing).entities
                    |> List.all (\e -> e.x > Core.gooseX playing + 100)
                    |> Expect.equal True
        ]
