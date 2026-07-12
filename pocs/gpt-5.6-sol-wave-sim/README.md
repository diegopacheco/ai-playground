# Pelagic Wave Simulator

Pelagic is a cinematic, browser-based ocean simulation rendered entirely with WebGL. It combines procedural water, sky reflections, foam, atmospheric depth, changing weather, and a fully rendered rubber duck that rides the wave surface.

![Pelagic wave simulator](screenshot.png)

## Run

```bash
./start.sh
```

Open `http://localhost:4173`.

```bash
./stop.sh
```

## Controls

- Wave height changes the vertical size of the swell.
- Wave length changes the distance between crests.
- Wind speed controls wave travel speed.
- Choppiness adds sharper high-frequency motion and foam.
- Storm front shifts the water and sky toward rougher, colder conditions.
- Calm, Swell, and Squall load tuned sea states.
- Live Motion pauses or resumes the simulation.
- Drag the ocean to orbit the camera and scroll to zoom.

## Design

The ocean is calculated in a WebGL fragment shader from layered directional waves. Surface normals are sampled from the same height field and used for Fresnel reflections, sunlight, fog, foam, and sky color. The duck is assembled from shaded 3D meshes and samples the same wave equation, giving it matching height, pitch, roll, and lateral drift.

The project has no runtime dependencies. A current browser with WebGL support and Python 3 for the local static server are required.
