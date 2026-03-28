import { useState, useEffect, useCallback } from "react";
import type {
  PrInfo,
  Counters,
  TestClassification,
  AgentAction,
  ConversationEntry,
  FileEntry,
  AgentLog,
} from "./types";
import {
  fetchStatus,
  fetchActions,
  fetchFiles,
  fetchLogs,
  fetchFileContent,
  fetchConversation,
  triggerRun,
} from "./api/client";
import { useSSE } from "./hooks/useSSE";
import Header from "./components/Header";
import CounterCards from "./components/CounterCards";
import ActivityLog from "./components/ActivityLog";
import FileExplorer from "./components/FileExplorer";
import FileViewer from "./components/FileViewer";
import AgentLogs from "./components/AgentLogs";
import PrConversation from "./components/PrConversation";

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [prInfo, setPrInfo] = useState<PrInfo | null>(null);
  const [counters, setCounters] = useState<Counters | null>(null);
  const [testClassification, setTestClassification] = useState<TestClassification | null>(null);
  const [actions, setActions] = useState<AgentAction[]>([]);
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [logs, setLogs] = useState<AgentLog[]>([]);
  const [conversation, setConversation] = useState<ConversationEntry[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [fileLoading, setFileLoading] = useState(false);
  const [agentRunning, setAgentRunning] = useState(false);

  const refreshAll = useCallback(() => {
    fetchStatus()
      .then((data) => {
        setPrInfo(data.pr_info);
        setCounters(data.counters);
        setTestClassification(data.test_classification);
      })
      .catch(() => {});
    fetchActions().then(setActions).catch(() => {});
    fetchFiles().then(setFiles).catch(() => {});
    fetchLogs().then(setLogs).catch(() => {});
    fetchConversation().then(setConversation).catch(() => {});
  }, []);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  const handleSelectFile = useCallback(async (path: string) => {
    setSelectedFile(path);
    setFileLoading(true);
    try {
      const data = await fetchFileContent(path);
      setFileContent(data.content);
    } catch {
      setFileContent("Failed to load file.");
    }
    setFileLoading(false);
  }, []);

  const handleRunAgent = useCallback(async () => {
    setAgentRunning(true);
    try {
      await triggerRun();
    } catch {
      setAgentRunning(false);
    }
  }, []);

  useSSE({
    onAction: useCallback(
      (action: AgentAction) => setActions((prev) => [...prev, action]),
      []
    ),
    onCounterUpdate: useCallback(
      (c: Counters) => setCounters(c),
      []
    ),
    onTestClassification: useCallback(
      (tc: TestClassification) => setTestClassification(tc),
      []
    ),
    onCycleEnd: useCallback(() => {
      refreshAll();
      setAgentRunning(false);
    }, [refreshAll]),
  });

  return (
    <div className="app">
      <Header
        prInfo={prInfo}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onRunAgent={handleRunAgent}
        running={agentRunning}
      />
      <main className="main-content">
        {activeTab === "dashboard" && (
          <div className="dashboard-layout">
            <CounterCards counters={counters} testClassification={testClassification} />
            <ActivityLog actions={actions} />
            <div className="explorer-viewer-layout">
              <FileExplorer
                files={files}
                selectedPath={selectedFile}
                onSelectFile={handleSelectFile}
              />
              <FileViewer
                path={selectedFile}
                content={fileContent}
                loading={fileLoading}
              />
            </div>
          </div>
        )}
        {activeTab === "conversation" && <PrConversation entries={conversation} actions={actions} />}
        {activeTab === "logs" && <AgentLogs logs={logs} />}
      </main>
    </div>
  );
}
