module InputTest exposing (suite)

import Expect
import Game.Input as Input exposing (Action(..))
import Json.Decode as Decode
import Test exposing (Test, describe, test)


decodeKey : String -> Bool -> Result Decode.Error Action
decodeKey key repeat =
    Decode.decodeString (Input.decoder Input.keyDown)
        ("{\"key\":\"" ++ key ++ "\",\"repeat\":" ++ (if repeat then "true" else "false") ++ "}")


suite : Test
suite =
    describe "Input"
        [ test "space jumps" <|
            \_ -> Input.keyDown " " |> Expect.equal (Just Jump)
        , test "releasing space ends the jump climb" <|
            \_ -> Input.keyUp " " |> Expect.equal (Just JumpReleased)
        , test "arrow down holds a duck and releasing stands up" <|
            \_ -> ( Input.keyDown "ArrowDown", Input.keyUp "ArrowDown" ) |> Expect.equal ( Just (Duck True), Just (Duck False) )
        , test "a held key auto-repeat is ignored so it cannot burn the double jump" <|
            \_ -> decodeKey " " True |> Result.toMaybe |> Expect.equal Nothing
        , test "a fresh key press is accepted" <|
            \_ -> decodeKey "p" False |> Expect.equal (Ok Pause)
        ]
