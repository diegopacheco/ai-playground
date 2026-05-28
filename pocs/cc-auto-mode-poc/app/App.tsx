import { useState } from "react";
import { StyleSheet, View } from "react-native";
import { StatusBar } from "expo-status-bar";
import { CameraScreen } from "./src/CameraScreen";
import { ResultOverlay } from "./src/ResultOverlay";
import { identifyObject } from "./src/openai";

type Phase = "ready" | "analyzing" | "result" | "error";

export default function App() {
  const [phase, setPhase] = useState<Phase>("ready");
  const [uri, setUri] = useState<string | null>(null);
  const [label, setLabel] = useState<string | undefined>();
  const [error, setError] = useState<string | undefined>();

  const onCapture = async (base64: string, capturedUri: string) => {
    setUri(capturedUri);
    setLabel(undefined);
    setError(undefined);
    setPhase("analyzing");
    try {
      const result = await identifyObject(base64);
      setLabel(result);
      setPhase("result");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      setPhase("error");
    }
  };

  const onRetake = () => {
    setUri(null);
    setLabel(undefined);
    setError(undefined);
    setPhase("ready");
  };

  return (
    <View style={styles.fill}>
      <StatusBar style="light" />
      {phase === "ready" || !uri ? (
        <CameraScreen onCapture={onCapture} />
      ) : (
        <ResultOverlay
          uri={uri}
          analyzing={phase === "analyzing"}
          label={label}
          error={error}
          onRetake={onRetake}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: "#000" },
});
