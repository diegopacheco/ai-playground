module Game.Config exposing
    ( acceleration
    , breadPerLife
    , districtLength
    , firstSpawn
    , flapVelocity
    , gooseScreenX
    , gravity
    , groundY
    , height
    , hurtTime
    , jumpCut
    , jumpVelocity
    , maxDelta
    , maxJumps
    , maxLives
    , maxSpeed
    , restartDelay
    , seagullSpeed
    , snap
    , startLives
    , startSpeed
    , stompBounce
    , width
    )


width : Float
width =
    320


height : Float
height =
    180


groundY : Float
groundY =
    150


gooseScreenX : Float
gooseScreenX =
    40


gravity : Float
gravity =
    900


jumpVelocity : Float
jumpVelocity =
    330


flapVelocity : Float
flapVelocity =
    270


jumpCut : Float
jumpCut =
    120


maxJumps : Int
maxJumps =
    2


startSpeed : Float
startSpeed =
    110


maxSpeed : Float
maxSpeed =
    260


acceleration : Float
acceleration =
    3


maxDelta : Float
maxDelta =
    0.05


startLives : Int
startLives =
    3


maxLives : Int
maxLives =
    5


breadPerLife : Int
breadPerLife =
    25


hurtTime : Float
hurtTime =
    1.5


stompBounce : Float
stompBounce =
    240


seagullSpeed : Float
seagullSpeed =
    35


snap : Float
snap =
    4


firstSpawn : Float
firstSpawn =
    360


districtLength : Float
districtLength =
    3000


restartDelay : Float
restartDelay =
    0.8
