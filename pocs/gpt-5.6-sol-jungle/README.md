# Ybyrá

Ybyrá is a full-screen journey through a living Brazilian rainforest. A calm Amazon tributary sits beneath shifting canopy light, passing rain, WebGL fireflies, hidden wildlife, and a locally generated ambient soundscape.

## Run

```bash
./start.sh
```

Open [http://127.0.0.1:8090](http://127.0.0.1:8090).

```bash
./stop.sh
```

## Explore

- Select “Begin listening” to wake the local soundscape
- Scroll continuously through river, high forest, rain, dusk, and night
- Select the coral snake, onça-pintada, and forest guardian when they appear
- Trigger or clear rain with the weather control
- Stop and start all forest motion
- Enter browser full-screen mode from the top-right control

## Design

The experience uses plain HTML, CSS, JavaScript, Web Audio, and WebGL with no external runtime libraries. WebGL renders moving atmospheric light and fireflies. Web Audio synthesizes river wash, rainfall, a low forest drone, and occasional bird calls in the browser. The rainforest artwork is original and stored locally.

## Controls

| Control | Action |
| --- | --- |
| Scroll | Continue the endless journey |
| Sound | Start or mute the soundscape |
| Weather | Start or stop rain |
| Stop | Freeze or resume motion |
| Full screen | Enter or leave browser full-screen mode |

Audio begins only after interaction because browsers require a user gesture before sound playback.
