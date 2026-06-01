import { useEffect } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { WIDTH, DEPTH, HEIGHT } from "../game/types";
import { Pit } from "./Pit";
import { ActivePiece } from "./ActivePiece";

function CameraRig() {
  const { camera } = useThree();
  useEffect(() => {
    camera.lookAt(0, 0, 0);
  }, [camera]);
  return null;
}

export function Scene() {
  return (
    <Canvas
      className="canvas"
      gl={{ alpha: true, antialias: true }}
      camera={{ position: [WIDTH * 1.9, HEIGHT * 0.35, DEPTH * 2.4], fov: 48 }}
    >
      <CameraRig />
      <ambientLight intensity={0.65} />
      <directionalLight position={[12, 18, 10]} intensity={1.15} />
      <directionalLight position={[-10, 4, -8]} intensity={0.4} color="#60a5fa" />
      <group position={[-WIDTH / 2, -HEIGHT / 2, -DEPTH / 2]}>
        <Pit />
        <ActivePiece />
      </group>
    </Canvas>
  );
}
