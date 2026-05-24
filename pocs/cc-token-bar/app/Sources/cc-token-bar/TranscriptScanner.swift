import Foundation

final class TranscriptScanner {
    private let root: URL
    private var cache: [String: (mtime: Date, session: SessionFile)] = [:]

    init() {
        self.root = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".claude/projects")
    }

    func scan() -> [SessionFile] {
        let fm = FileManager.default
        guard let projectDirs = try? fm.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        var results: [SessionFile] = []
        var seenPaths = Set<String>()

        for pdir in projectDirs {
            var isDir: ObjCBool = false
            fm.fileExists(atPath: pdir.path, isDirectory: &isDir)
            guard isDir.boolValue else { continue }
            let projectPath = decodeProjectPath(pdir.lastPathComponent)
            guard let files = try? fm.contentsOfDirectory(
                at: pdir,
                includingPropertiesForKeys: [.contentModificationDateKey]
            ) else { continue }

            for url in files where url.pathExtension == "jsonl" {
                seenPaths.insert(url.path)
                let attrs = try? url.resourceValues(forKeys: [.contentModificationDateKey])
                let mtime = attrs?.contentModificationDate ?? .distantPast
                if let c = cache[url.path], c.mtime == mtime {
                    results.append(c.session)
                    continue
                }
                let sid = url.deletingPathExtension().lastPathComponent
                if let parsed = parse(url: url, sessionId: sid, projectPath: projectPath, fileMtime: mtime) {
                    cache[url.path] = (mtime, parsed)
                    results.append(parsed)
                }
            }
        }

        for key in cache.keys where !seenPaths.contains(key) {
            cache.removeValue(forKey: key)
        }
        return results
    }

    private func decodeProjectPath(_ dirName: String) -> String {
        return "/" + dirName.replacingOccurrences(of: "-", with: "/")
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    private func parse(url: URL, sessionId: String, projectPath: String, fileMtime: Date) -> SessionFile? {
        guard let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8) else { return nil }

        var byModel: [String: ModelUsage] = [:]
        var minTs: String?
        var maxTs: String?

        text.enumerateLines { line, _ in
            guard let lineData = line.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any] else { return }
            if let ts = obj["timestamp"] as? String {
                if minTs == nil || ts < minTs! { minTs = ts }
                if maxTs == nil || ts > maxTs! { maxTs = ts }
            }
            guard let message = obj["message"] as? [String: Any],
                  let usage = message["usage"] as? [String: Any] else { return }
            let model = (message["model"] as? String) ?? "unknown"
            let input = usage["input_tokens"] as? Int ?? 0
            let output = usage["output_tokens"] as? Int ?? 0
            let cw = usage["cache_creation_input_tokens"] as? Int ?? 0
            let cr = usage["cache_read_input_tokens"] as? Int ?? 0
            var ws = 0
            var wf = 0
            if let stu = usage["server_tool_use"] as? [String: Any] {
                ws = stu["web_search_requests"] as? Int ?? 0
                wf = stu["web_fetch_requests"] as? Int ?? 0
            }
            var m = byModel[model] ?? ModelUsage(
                input_tokens: 0, output_tokens: 0,
                cache_creation_input_tokens: 0, cache_read_input_tokens: 0,
                web_search_requests: 0, web_fetch_requests: 0, messages: 0)
            m.input_tokens += input
            m.output_tokens += output
            m.cache_creation_input_tokens += cw
            m.cache_read_input_tokens += cr
            m.web_search_requests += ws
            m.web_fetch_requests += wf
            m.messages += 1
            byModel[model] = m
        }

        let fallback = TranscriptScanner.iso.string(from: fileMtime)
        return SessionFile(
            session_id: sessionId,
            project_path: projectPath,
            started_at: minTs ?? fallback,
            updated_at: maxTs ?? fallback,
            by_model: byModel
        )
    }

    private static let iso: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()
}
