module WorldTest exposing (suite)

import Expect
import Game.Config as Config
import Game.District as District exposing (District(..))
import Game.Spawner as Spawner
import Random
import Test exposing (Test, describe, test)
import View.Sprites as Sprites


suite : Test
suite =
    describe "World"
        [ test "the same seed builds the same street so runs are reproducible" <|
            \_ ->
                Spawner.generate 400 110 (Random.initialSeed 7)
                    |> (\( entities, next, _ ) -> ( entities, next ))
                    |> Expect.equal
                        (Spawner.generate 400 110 (Random.initialSeed 7) |> (\( entities, next, _ ) -> ( entities, next )))
        , test "the next pattern always starts after a gap to leave time to react" <|
            \_ ->
                List.range 1 200
                    |> List.all
                        (\s ->
                            let
                                ( entities, next, _ ) =
                                    Spawner.generate 0 Config.startSpeed (Random.initialSeed s)

                                furthest =
                                    entities |> List.map .x |> List.maximum |> Maybe.withDefault 0
                            in
                            next - furthest >= 80
                        )
                    |> Expect.equal True
        , test "the run tours Mission, Chinatown and Golden Gate then loops" <|
            \_ ->
                List.map (\i -> District.at (toFloat i * Config.districtLength + 1)) [ 0, 1, 2, 3 ]
                    |> Expect.equal [ Mission, Chinatown, GoldenGate, Mission ]
        , test "every sprite is a clean rectangle so the pixel art does not skew" <|
            \_ ->
                Sprites.all
                    |> List.all
                        (\rows ->
                            case rows of
                                first :: rest ->
                                    List.all (\r -> String.length r == String.length first) rest

                                [] ->
                                    False
                        )
                    |> Expect.equal True
        ]
