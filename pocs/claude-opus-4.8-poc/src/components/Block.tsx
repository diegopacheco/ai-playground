import * as THREE from "three";

const BOX = new THREE.BoxGeometry(0.9, 0.9, 0.9);
const EDGES = new THREE.EdgesGeometry(BOX);

export function Block({
  position,
  color,
  opacity = 1,
  emissive = 0,
  edge = "#0b1020",
}: {
  position: [number, number, number];
  color: string;
  opacity?: number;
  emissive?: number;
  edge?: string;
}) {
  return (
    <group position={position}>
      <mesh geometry={BOX}>
        <meshStandardMaterial
          color={color}
          transparent={opacity < 1}
          opacity={opacity}
          emissive={color}
          emissiveIntensity={emissive}
          metalness={0.25}
          roughness={0.45}
        />
      </mesh>
      <lineSegments geometry={EDGES}>
        <lineBasicMaterial color={edge} transparent={opacity < 1} opacity={opacity} />
      </lineSegments>
    </group>
  );
}
