module Game.Core exposing
    ( Game
    , Screen(..)
    , apply
    , gooseX
    , init
    , score
    , start
    , step
    )

import Game.Collision as Collision
import Game.Config as Config
import Game.Entity as Entity exposing (Entity, Kind(..))
import Game.Goose as Goose exposing (Goose)
import Game.Input exposing (Action(..))
import Game.Spawner as Spawner
import Random


type Screen
    = Title
    | Playing
    | Paused
    | GameOver


type alias Game =
    { screen : Screen
    , goose : Goose
    , entities : List Entity
    , distance : Float
    , speed : Float
    , bread : Int
    , stomps : Int
    , lives : Int
    , best : Int
    , seed : Random.Seed
    , nextSpawn : Float
    , duckHeld : Bool
    , clock : Float
    }


init : Int -> Game
init seedValue =
    { screen = Title
    , goose = Goose.init
    , entities = []
    , distance = 0
    , speed = Config.startSpeed
    , bread = 0
    , stomps = 0
    , lives = Config.startLives
    , best = 0
    , seed = Random.initialSeed seedValue
    , nextSpawn = Config.firstSpawn
    , duckHeld = False
    , clock = 0
    }


start : Game -> Game
start game =
    let
        fresh =
            init 0
    in
    { fresh | screen = Playing, best = game.best, seed = game.seed }


score : Game -> Int
score game =
    floor (game.distance / 10) + game.bread * 10 + game.stomps * 50


gooseX : Game -> Float
gooseX game =
    game.distance + Config.gooseScreenX


apply : Action -> Game -> Game
apply action game =
    case ( action, game.screen ) of
        ( Duck held, _ ) ->
            { game | duckHeld = held }

        ( Jump, Playing ) ->
            { game | goose = Goose.jump game.goose }

        ( JumpReleased, Playing ) ->
            { game | goose = Goose.release game.goose }

        ( Pause, Playing ) ->
            { game | screen = Paused }

        ( Pause, Paused ) ->
            { game | screen = Playing }

        ( Confirm, Paused ) ->
            { game | screen = Playing }

        ( Jump, Title ) ->
            start game

        ( Confirm, Title ) ->
            start game

        ( Jump, GameOver ) ->
            restart game

        ( Confirm, GameOver ) ->
            restart game

        _ ->
            game


restart : Game -> Game
restart game =
    if game.clock >= Config.restartDelay then
        start game

    else
        game


step : Float -> Game -> Game
step rawDt game =
    let
        dt =
            min Config.maxDelta rawDt
    in
    case game.screen of
        Playing ->
            { game | clock = game.clock + dt }
                |> advance dt
                |> spawn
                |> moveGoose dt
                |> resolve
                |> cleanup

        Paused ->
            game

        _ ->
            { game | clock = game.clock + dt }


advance : Float -> Game -> Game
advance dt game =
    { game
        | distance = game.distance + game.speed * dt
        , speed = min Config.maxSpeed (game.speed + Config.acceleration * dt)
        , entities = List.map (Entity.move dt) game.entities
    }


spawn : Game -> Game
spawn game =
    if game.nextSpawn < game.distance + Config.width + 40 then
        let
            ( entities, nextSpawn, seed ) =
                Spawner.generate game.nextSpawn game.speed game.seed
        in
        { game | entities = game.entities ++ entities, nextSpawn = nextSpawn, seed = seed }

    else
        game


moveGoose : Float -> Game -> Game
moveGoose dt game =
    { game
        | goose =
            game.goose
                |> Goose.step dt (floorUnder game)
                |> Goose.setDuck game.duckHeld
    }


floorUnder : Game -> Float
floorUnder game =
    let
        gooseBox =
            Goose.box (gooseX game) game.goose
    in
    game.entities
        |> List.filter (\entity -> entity.kind == CableCar)
        |> List.map Entity.box
        |> List.filter (\roof -> Collision.overlapsX gooseBox roof && game.goose.y >= Collision.top roof - Config.snap)
        |> List.map Collision.top
        |> List.maximum
        |> Maybe.withDefault 0


resolve : Game -> Game
resolve game =
    let
        ( resolved, kept ) =
            List.foldl touch ( { game | entities = [] }, [] ) game.entities
    in
    { resolved | entities = List.reverse kept }


touch : Entity -> ( Game, List Entity ) -> ( Game, List Entity )
touch entity ( game, kept ) =
    let
        entityBox =
            Entity.box entity

        above =
            game.goose.y >= Collision.top entityBox - Config.snap
    in
    if game.screen /= Playing || not (Collision.overlaps (Goose.box (gooseX game) game.goose) entityBox) then
        ( game, entity :: kept )

    else
        case entity.kind of
            Sourdough ->
                ( collect game, kept )

            Seagull ->
                if game.goose.vy < 0 && above then
                    ( stomp game, kept )

                else
                    ( hit game, entity :: kept )

            CableCar ->
                if above then
                    ( game, entity :: kept )

                else
                    ( hit game, entity :: kept )

            _ ->
                ( hit game, entity :: kept )


collect : Game -> Game
collect game =
    let
        bread =
            game.bread + 1

        lives =
            if modBy Config.breadPerLife bread == 0 then
                min Config.maxLives (game.lives + 1)

            else
                game.lives
    in
    { game | bread = bread, lives = lives }


stomp : Game -> Game
stomp game =
    { game | stomps = game.stomps + 1, goose = Goose.bounce game.goose }


hit : Game -> Game
hit game =
    if Goose.isInvincible game.goose then
        game

    else if game.lives <= 1 then
        { game
            | lives = 0
            , screen = GameOver
            , clock = 0
            , best = max game.best (score game)
        }

    else
        { game | lives = game.lives - 1, goose = Goose.hurt game.goose }


cleanup : Game -> Game
cleanup game =
    { game | entities = List.filter (\entity -> entity.x + (Entity.size entity.kind).w > game.distance - 20) game.entities }
