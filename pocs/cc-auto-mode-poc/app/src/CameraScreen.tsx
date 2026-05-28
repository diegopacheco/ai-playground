import { useRef } from "react";
import { StyleSheet, View, Text, TouchableOpacity } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";

type Props = {
  onCapture: (base64: string, uri: string) => void;
};

export function CameraScreen({ onCapture }: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);

  if (!permission) {
    return <View style={styles.center} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.message}>Camera access is needed to point and tell.</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const capture = async () => {
    const photo = await cameraRef.current?.takePictureAsync({
      base64: true,
      quality: 0.5,
    });
    if (photo?.base64 && photo?.uri) {
      onCapture(photo.base64, photo.uri);
    }
  };

  return (
    <View style={styles.fill}>
      <CameraView ref={cameraRef} style={styles.fill} facing="back" />
      <View style={styles.shutterBar}>
        <TouchableOpacity style={styles.shutter} onPress={capture} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#000",
    padding: 24,
  },
  message: { color: "#fff", fontSize: 16, textAlign: "center", marginBottom: 16 },
  button: { backgroundColor: "#fff", paddingHorizontal: 20, paddingVertical: 12, borderRadius: 8 },
  buttonText: { color: "#000", fontSize: 16, fontWeight: "600" },
  shutterBar: {
    position: "absolute",
    bottom: 48,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  shutter: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: "#fff",
    borderWidth: 4,
    borderColor: "rgba(255,255,255,0.5)",
  },
});
