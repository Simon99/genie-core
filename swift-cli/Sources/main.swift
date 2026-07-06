import Foundation
import Speech

// MARK: - Data Types

struct Segment: Codable {
    let start: Double
    let end: Double
    let text: String
}

struct TranscribeRequest: Codable {
    let path: String
    let language: String?
}

struct TranscribeResponse: Codable {
    let segments: [Segment]
}

enum TranscribeOutcome {
    case success([Segment])
    case failure(String)
}

// MARK: - Speech Recognition

func transcribe(filePath: String, language: String) -> TranscribeOutcome {
    let semaphore = DispatchSemaphore(value: 0)
    let url = URL(fileURLWithPath: filePath)
    let locale = Locale(identifier: language)

    guard let recognizer = SFSpeechRecognizer(locale: locale) else {
        return .failure("Speech recognizer not available for locale \(language)")
    }

    guard recognizer.isAvailable else {
        return .failure("Speech recognizer is not available (locale \(language))")
    }

    let request = SFSpeechURLRecognitionRequest(url: url)
    request.shouldReportPartialResults = false
    request.addsPunctuation = true
    // On-device recognition avoids the ~60s limit of server-based recognition.
    if recognizer.supportsOnDeviceRecognition {
        request.requiresOnDeviceRecognition = true
    }

    var allSegments: [Segment] = []
    var errorMessage: String? = nil

    let task = recognizer.recognitionTask(with: request) { result, error in
        if let error = error {
            errorMessage = error.localizedDescription
            semaphore.signal()
            return
        }

        guard let result = result else {
            semaphore.signal()
            return
        }

        if result.isFinal {
            for segment in result.bestTranscription.segments {
                let seg = Segment(
                    start: segment.timestamp,
                    end: segment.timestamp + segment.duration,
                    text: segment.substring
                )
                allSegments.append(seg)
            }
            semaphore.signal()
        }
    }

    if semaphore.wait(timeout: .now() + 900) == .timedOut {
        task.cancel()
        return .failure("Speech recognition timed out after 900 seconds")
    }

    if let message = errorMessage {
        return .failure(message)
    }

    return .success(mergeSegments(allSegments, maxGap: 1.0, maxDuration: 10.0))
}

func mergeSegments(_ segments: [Segment], maxGap: Double, maxDuration: Double) -> [Segment] {
    guard !segments.isEmpty else { return [] }

    var merged: [Segment] = []
    var currentStart = segments[0].start
    var currentEnd = segments[0].end
    var currentText = segments[0].text

    for i in 1..<segments.count {
        let seg = segments[i]
        let gap = seg.start - currentEnd
        let wouldBeDuration = seg.end - currentStart

        if gap <= maxGap && wouldBeDuration <= maxDuration {
            currentEnd = seg.end
            currentText += seg.text
        } else {
            merged.append(Segment(start: currentStart, end: currentEnd, text: currentText))
            currentStart = seg.start
            currentEnd = seg.end
            currentText = seg.text
        }
    }
    merged.append(Segment(start: currentStart, end: currentEnd, text: currentText))

    return merged
}

// MARK: - HTTP Server (minimal, no dependencies)

class HTTPServer {
    let port: UInt16
    var socket: Int32 = -1
    let clientQueue = DispatchQueue(label: "com.genie.speech-proxy.clients", attributes: .concurrent)

    init(port: UInt16) {
        self.port = port
    }

    func start() {
        socket = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard socket >= 0 else {
            fputs("Error: socket() failed: \(String(cString: strerror(errno)))\n", stderr)
            exit(1)
        }
        var yes: Int32 = 1
        setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                bind(socket, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            fputs("Error: bind() failed on 127.0.0.1:\(port): \(String(cString: strerror(errno)))\n", stderr)
            exit(1)
        }

        guard listen(socket, 5) == 0 else {
            fputs("Error: listen() failed on port \(port): \(String(cString: strerror(errno)))\n", stderr)
            exit(1)
        }
        fputs("Speech proxy listening on http://127.0.0.1:\(port)\n", stderr)

        while true {
            let client = accept(socket, nil, nil)
            if client < 0 { continue }
            clientQueue.async { [weak self] in
                self?.handleClient(client)
            }
        }
    }

    /// Read a full HTTP request: headers, then body until Content-Length is satisfied.
    func readRequest(_ client: Int32) -> Data? {
        let headerTerminator = Data("\r\n\r\n".utf8)
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 65536)

        // Read until the header terminator arrives (with a 1 MiB safety cap)
        var headerRange = data.range(of: headerTerminator)
        while headerRange == nil {
            let n = read(client, &buffer, buffer.count)
            if n <= 0 {
                return data.isEmpty ? nil : data
            }
            data.append(contentsOf: buffer[0..<n])
            headerRange = data.range(of: headerTerminator)
            if headerRange == nil && data.count > 1_048_576 {
                return data
            }
        }

        guard let headers = headerRange else { return data }
        let contentLength = parseContentLength(data.subdata(in: data.startIndex..<headers.lowerBound))
        var bodyBytes = data.endIndex - headers.upperBound
        while bodyBytes < contentLength {
            let n = read(client, &buffer, buffer.count)
            if n <= 0 { break }
            data.append(contentsOf: buffer[0..<n])
            bodyBytes += n
        }
        return data
    }

    func parseContentLength(_ headerData: Data) -> Int {
        guard let headerStr = String(data: headerData, encoding: .utf8) else { return 0 }
        for line in headerStr.split(separator: "\r\n") {
            let parts = line.split(separator: ":", maxSplits: 1)
            if parts.count == 2 &&
               parts[0].trimmingCharacters(in: .whitespaces).lowercased() == "content-length" {
                return Int(parts[1].trimmingCharacters(in: .whitespaces)) ?? 0
            }
        }
        return 0
    }

    func handleClient(_ client: Int32) {
        guard let requestData = readRequest(client) else {
            close(client)
            return
        }

        let headerTerminator = Data("\r\n\r\n".utf8)
        guard let headerRange = requestData.range(of: headerTerminator) else {
            respond(client, status: 400, body: "{\"error\":\"malformed request\"}")
            return
        }

        let headerData = requestData.subdata(in: requestData.startIndex..<headerRange.lowerBound)
        guard let headerStr = String(data: headerData, encoding: .utf8),
              let firstLine = headerStr.split(separator: "\r\n").first else {
            respond(client, status: 400, body: "{\"error\":\"malformed request\"}")
            return
        }

        let parts = firstLine.split(separator: " ")
        guard parts.count >= 2 else {
            respond(client, status: 400, body: "{\"error\":\"malformed request line\"}")
            return
        }
        let method = String(parts[0])
        let path = String(parts[1])

        if method == "GET" && path == "/health" {
            respond(client, status: 200, body: "{\"status\":\"ok\"}")
            return
        }

        if method == "POST" && path == "/transcribe" {
            let bodyData = requestData.subdata(in: headerRange.upperBound..<requestData.endIndex)
            guard let body = String(data: bodyData, encoding: .utf8), !body.isEmpty else {
                respond(client, status: 400, body: "{\"error\":\"no body\"}")
                return
            }
            handleTranscribe(client, body: body)
            return
        }

        respond(client, status: 404, body: "{\"error\":\"not found\"}")
    }

    func handleTranscribe(_ client: Int32, body: String) {
        guard let data = body.data(using: .utf8),
              let req = try? JSONDecoder().decode(TranscribeRequest.self, from: data) else {
            respond(client, status: 400, body: "{\"error\":\"invalid json\"}")
            return
        }

        let lang = req.language ?? "zh-Hans"
        fputs("Transcribing: \(req.path) (lang: \(lang))\n", stderr)

        switch transcribe(filePath: req.path, language: lang) {
        case .failure(let message):
            if let jsonData = try? JSONEncoder().encode(["error": message]),
               let jsonStr = String(data: jsonData, encoding: .utf8) {
                respond(client, status: 500, body: jsonStr)
            } else {
                respond(client, status: 500, body: "{\"error\":\"transcription failed\"}")
            }
        case .success(let segments):
            let response = TranscribeResponse(segments: segments)
            if let jsonData = try? JSONEncoder().encode(response),
               let jsonStr = String(data: jsonData, encoding: .utf8) {
                respond(client, status: 200, body: jsonStr)
            } else {
                respond(client, status: 500, body: "{\"error\":\"encoding failed\"}")
            }
        }
    }

    func respond(_ client: Int32, status: Int, body: String) {
        let statusText = status == 200 ? "OK" : "Error"
        let response = "HTTP/1.1 \(status) \(statusText)\r\nContent-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\nConnection: close\r\n\r\n\(body)"
        let bytes = [UInt8](response.utf8)
        var sent = 0
        while sent < bytes.count {
            let n = bytes.withUnsafeBufferPointer { buf in
                write(client, buf.baseAddress! + sent, bytes.count - sent)
            }
            if n <= 0 { break }
            sent += n
        }
        close(client)
    }
}

// MARK: - Main

let args = CommandLine.arguments

if args.contains("--server") {
    var port: UInt16 = 5300
    if let flagIdx = args.firstIndex(of: "--port") {
        let valueIdx = args.index(after: flagIdx)
        if valueIdx < args.count, let parsed = UInt16(args[valueIdx]) {
            port = parsed
        } else {
            fputs("Error: --port requires a numeric argument (1-65535)\n", stderr)
            exit(1)
        }
    }

    SFSpeechRecognizer.requestAuthorization { status in
        switch status {
        case .authorized:
            let server = HTTPServer(port: port)
            server.start()
        default:
            // Deliberate exit(0): the LaunchAgent plist uses KeepAlive/SuccessfulExit=false,
            // which relaunches only on NON-zero exit. A zero exit here prevents an infinite
            // restart + permission-prompt loop when speech recognition is not yet authorized.
            // Re-grant by answering the system prompt or running once in a GUI session.
            fputs("Error: Speech recognition not authorized (status: \(status.rawValue)). "
                + "Grant permission in System Settings or run once in a GUI session, "
                + "then: launchctl kickstart -k gui/$UID/com.genie.speech-proxy\n", stderr)
            exit(0)
        }
    }
    RunLoop.main.run()
} else {
    // CLI mode (original behavior)
    var input = ""
    var language = "zh-Hans"

    var i = 1
    while i < args.count {
        switch args[i] {
        case "--input":
            i += 1
            if i < args.count { input = args[i] }
        case "--language":
            i += 1
            if i < args.count { language = args[i] }
        default:
            break
        }
        i += 1
    }

    guard !input.isEmpty else {
        fputs("Usage:\n  genie-speech-cli --input <file> [--language <locale>]\n  genie-speech-cli --server [--port 5300]\n", stderr)
        exit(1)
    }

    guard FileManager.default.fileExists(atPath: input) else {
        fputs("Error: File not found: \(input)\n", stderr)
        exit(1)
    }

    SFSpeechRecognizer.requestAuthorization { status in
        switch status {
        case .authorized:
            switch transcribe(filePath: input, language: language) {
            case .failure(let message):
                fputs("Error: \(message)\n", stderr)
                exit(1)
            case .success(let segments):
                let encoder = JSONEncoder()
                encoder.outputFormatting = .prettyPrinted
                if let data = try? encoder.encode(segments) {
                    print(String(data: data, encoding: .utf8) ?? "[]")
                }
                exit(0)
            }
        default:
            fputs("Error: Speech recognition not authorized (status: \(status.rawValue))\n", stderr)
            exit(1)
        }
    }
    RunLoop.main.run(until: Date(timeIntervalSinceNow: 1800))
}
