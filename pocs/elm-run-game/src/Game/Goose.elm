module Game.Goose exposing
    ( Goose
    , bounce
    , box
    , hurt
    , init
    , isInvincible
    , jump
    , release
    , setDuck
    , step
    )

import Game.Collision exposing (Box)
import Game.Config as Config


type alias Goose =
    { y : Float
    , vy : Float
    , jumps : Int
    , grounded : Bool
    , ducking : Bool
    , hurtTimer : Float
    , runClock : Float
    }


init : Goose
init =
    { y = 0
    , vy = 0
    , jumps = 0
    , grounded = True
    , ducking = False
    , hurtTimer = 0
    , runClock = 0
    }


jump : Goose -> Goose
jump goose =
    if goose.jumps >= Config.maxJumps then
        goose

    else
        { goose
            | vy = jumpPower goose.jumps
            , jumps = goose.jumps + 1
            , grounded = False
            , ducking = False
        }


jumpPower : Int -> Float
jumpPower jumps =
    if jumps == 0 then
        Config.jumpVelocity

    else
        Config.flapVelocity


release : Goose -> Goose
release goose =
    { goose | vy = min goose.vy Config.jumpCut }


setDuck : Bool -> Goose -> Goose
setDuck held goose =
    { goose | ducking = held && goose.grounded }


step : Float -> Float -> Goose -> Goose
step dt floor goose =
    let
        vy =
            goose.vy - Config.gravity * dt

        y =
            goose.y + vy * dt

        ticked =
            { goose
                | hurtTimer = max 0 (goose.hurtTimer - dt)
                , runClock = goose.runClock + dt
            }
    in
    if y <= floor && vy <= 0 then
        { ticked | y = floor, vy = 0, jumps = 0, grounded = True }

    else
        { ticked | y = y, vy = vy, grounded = False, jumps = airJumps goose }


airJumps : Goose -> Int
airJumps goose =
    if goose.grounded then
        max 1 goose.jumps

    else
        goose.jumps


bounce : Goose -> Goose
bounce goose =
    { goose | vy = Config.stompBounce, jumps = 1, grounded = False }


hurt : Goose -> Goose
hurt goose =
    { goose | hurtTimer = Config.hurtTime }


isInvincible : Goose -> Bool
isInvincible goose =
    goose.hurtTimer > 0


box : Float -> Goose -> Box
box worldX goose =
    { x = worldX + 3
    , y = goose.y
    , w = 11
    , h =
        if goose.ducking then
            8

        else
            14
    }
