module Game.Input exposing (Action(..), decoder, keyDown, keyUp)

import Json.Decode as Decode exposing (Decoder)


type Action
    = Jump
    | JumpReleased
    | Duck Bool
    | Pause
    | Confirm


keyDown : String -> Maybe Action
keyDown key =
    if isJumpKey key then
        Just Jump

    else if isDuckKey key then
        Just (Duck True)

    else if List.member key [ "p", "P", "Escape" ] then
        Just Pause

    else if key == "Enter" then
        Just Confirm

    else
        Nothing


keyUp : String -> Maybe Action
keyUp key =
    if isJumpKey key then
        Just JumpReleased

    else if isDuckKey key then
        Just (Duck False)

    else
        Nothing


isJumpKey : String -> Bool
isJumpKey key =
    List.member key [ " ", "ArrowUp", "w", "W" ]


isDuckKey : String -> Bool
isDuckKey key =
    List.member key [ "ArrowDown", "s", "S" ]


decoder : (String -> Maybe Action) -> Decoder Action
decoder toAction =
    Decode.map2 Tuple.pair
        (Decode.field "key" Decode.string)
        (Decode.oneOf [ Decode.field "repeat" Decode.bool, Decode.succeed False ])
        |> Decode.andThen (accept toAction)


accept : (String -> Maybe Action) -> ( String, Bool ) -> Decoder Action
accept toAction ( key, repeat ) =
    case ( repeat, toAction key ) of
        ( False, Just action ) ->
            Decode.succeed action

        _ ->
            Decode.fail "ignored key"
