# Ybyrá

Ybyrá is a full-screen journey through a living Brazilian rainforest. A calm Amazon tributary sits beneath shifting canopy light, passing rain, realistic hidden wildlife, and a locally generated ambient soundscape.

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
- Scroll continuously through seven connected rainforest views with cipó, samambaia, old woods, wet canopy, roots, and late light
- Select the coral snake, onça-pintada, forest guardian, and flecheira when they appear
- Trigger or clear rain with the weather control
- Stop and start all forest motion
- Enter browser full-screen mode from the top-right control

## Design

The experience uses plain HTML, CSS, JavaScript, Web Audio, and Canvas 2D with no external runtime libraries. Canvas 2D creates randomized rainfall with varied depth and speed. Web Audio synthesizes river wash, moving leaves, cicadas, varied spatial bird calls, distant frogs, and rainfall in the browser. The seven rainforest views and wildlife artwork are original and stored locally.

## Controls

| Control | Action |
| --- | --- |
| Scroll | Continue the endless journey |
| Sound | Start or mute the soundscape |
| Weather | Start or stop rain |
| Stop | Freeze or resume motion |
| Full screen | Enter or leave browser full-screen mode |

Audio begins only after interaction because browsers require a user gesture before sound playback.
