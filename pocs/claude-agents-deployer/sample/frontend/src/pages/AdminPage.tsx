import { useEffect, useState } from "react";
import { useSettings, useUpdateSettings } from "../hooks/useApi";
import type { AppSettings } from "../types";

export default function AdminPage() {
  const {
    data: settings,
    isLoading,
    error,
    refetch,
    isFetching,
  } = useSettings();
  const updateSettings = useUpdateSettings();
  const [commentsEnabled, setCommentsEnabled] = useState(true);
  const [backgroundTheme, setBackgroundTheme] =
    useState<AppSettings["backgroundTheme"]>("classic");

  useEffect(() => {
    if (settings) {
      setCommentsEnabled(settings.commentsEnabled);
      setBackgroundTheme(settings.backgroundTheme);
    }
  }, [settings]);

  function handleSave(e: React.FormEvent) {
    e.preventDefault();
    updateSettings.mutate({ commentsEnabled, backgroundTheme });
  }

  if (isLoading) {
    return (
      <div className="flex justify-center py-20">
        <div className="text-gray-500 text-lg">Loading settings...</div>
      </div>
    );
  }

  const loadErrorMessage = error instanceof Error ? error.message : "";

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 mb-8">Admin Panel</h1>
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 mb-6">
          <div>Failed to load admin settings.</div>
          {loadErrorMessage && <div className="text-sm mt-1">{loadErrorMessage}</div>}
          <button
            type="button"
            onClick={() => refetch()}
            disabled={isFetching}
            className="mt-3 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 disabled:opacity-50"
          >
            {isFetching ? "Retrying..." : "Retry"}
          </button>
        </div>
      )}
      <form
        onSubmit={handleSave}
        className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 space-y-8"
      >
        <div className="space-y-3">
          <h2 className="text-lg font-semibold text-gray-900">Comments</h2>
          <label className="flex items-center gap-3 text-gray-700">
            <input
              type="checkbox"
              checked={commentsEnabled}
              onChange={(e) => setCommentsEnabled(e.target.checked)}
              className="h-4 w-4"
            />
            Enable comments across all posts
          </label>
        </div>

        <div className="space-y-3">
          <h2 className="text-lg font-semibold text-gray-900">Background Theme</h2>
          <select
            value={backgroundTheme}
            onChange={(e) =>
              setBackgroundTheme(e.target.value as AppSettings["backgroundTheme"])
            }
            className="w-full border border-gray-300 rounded-lg px-4 py-2"
          >
            <option value="classic">Classic</option>
            <option value="forest">Forest</option>
            <option value="sunset">Sunset</option>
          </select>
        </div>

        <button
          type="submit"
          disabled={updateSettings.isPending}
          className="bg-indigo-600 text-white px-6 py-2.5 rounded-lg hover:bg-indigo-700 transition-colors font-medium disabled:opacity-50"
        >
          {updateSettings.isPending ? "Saving..." : "Save Settings"}
        </button>
        {updateSettings.isSuccess && (
          <p className="text-green-700">Settings saved.</p>
        )}
        {updateSettings.isError && (
          <p className="text-red-600">Failed to save settings.</p>
        )}
      </form>
    </div>
  );
}
