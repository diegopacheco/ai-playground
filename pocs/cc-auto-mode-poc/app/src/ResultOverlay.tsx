import { StyleSheet, View, Text, Image, TouchableOpacity, ActivityIndicator } from "react-native";

type Props = {
  uri: string;
  analyzing: boolean;
  label?: string;
  error?: string;
  onRetake: () => void;
};

export function ResultOverlay({ uri, analyzing, label, error, onRetake }: Props) {
  return (
    <View style={styles.fill}>
      <Image source={{ uri }} style={styles.fill} resizeMode="cover" />

      {analyzing && (
        <View style={styles.banner}>
          <ActivityIndicator color="#fff" />
          <Text style={styles.bannerText}>Analyzing…</Text>
        </View>
      )}

      {!analyzing && label && (
        <View style={styles.banner}>
          <Text style={styles.label}>{label}</Text>
        </View>
      )}

      {!analyzing && error && (
        <View style={styles.banner}>
          <Text style={styles.error}>{error}</Text>
        </View>
      )}

      {!analyzing && (
        <View style={styles.shutterBar}>
          <TouchableOpacity style={styles.button} onPress={onRetake}>
            <Text style={styles.buttonText}>{error ? "Try again" : "Retake"}</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
  banner: {
    position: "absolute",
    top: 64,
    left: 16,
    right: 16,
    backgroundColor: "rgba(0,0,0,0.7)",
    borderRadius: 12,
    padding: 16,
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  bannerText: { color: "#fff", fontSize: 16 },
  label: { color: "#fff", fontSize: 20, fontWeight: "600" },
  error: { color: "#ff9f9f", fontSize: 16 },
  shutterBar: { position: "absolute", bottom: 48, left: 0, right: 0, alignItems: "center" },
  button: { backgroundColor: "#fff", paddingHorizontal: 28, paddingVertical: 14, borderRadius: 10 },
  buttonText: { color: "#000", fontSize: 16, fontWeight: "600" },
});
