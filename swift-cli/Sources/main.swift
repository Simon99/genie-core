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

// MARK: - Speech Recognition

func transcribe(filePath: String, language: String) -> [Segment] {
    let semaphore = DispatchSemaphore(value: 0)
    let url = URL(fileURLWithPath: filePath)
    let locale = Locale(identifier: language)

    guard let recognizer = SFSpeechRecognizer(locale: locale) else {
        fputs("Error: Speech recognizer not available for locale \(language)\n", stderr)
        return []
    }

    guard recognizer.isAvailable else {
        fputs("Error: Speech recognizer is not available\n", stderr)
        return []
    }

    let request = SFSpeechURLRecognitionRequest(url: url)
    request.shouldReportPartialResults = false
    request.addsPunctuation = true

    var allSegments: [Segment] = []

    recognizer.recognitionTask(with: request) { result, error in
        if let error = error {
            fputs("Error: \(error.localizedDescription)\n", stderr)
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

    semaphore.wait()
    return mergeSegments(allSegments, maxGap: 1.0, maxDuration: 10.0)
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

    init(port: UInt16) {
        self.port = port
    }

    func start() {
        socket = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        var yes: Int32 = 1
        setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = INADDR_ANY

        withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                bind(socket, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }

        listen(socket, 5)
        fputs("Speech proxy listening on http://localhost:\(port)\n", stderr)

        while true {
            let client = accept(socket, nil, nil)
            if client < 0 { continue }
            handleClient(client)
        }
    }

    func handleClient(_ client: Int32) {
        var buffer = [UInt8](repeating: 0, count: 65536)
        let n = read(client, &buffer, buffer.count)
        guard n > 0 else { close(client); return }

        let request = String(bytes: buffer[0..<n], encoding: .utf8) ?? ""
        let lines = request.split(separator: "\r\n")
        guard let firstLine = lines.first else { close(client); return }

        let parts = firstLine.split(separator: " ")
        guard parts.count >= 2 else { close(client); return }
        let method = String(parts[0])
        let path = String(parts[1])

        if method == "GET" && path == "/health" {
            respond(client, status: 200, body: "{\"status\":\"ok\"}")
            return
        }

        if method == "POST" && path == "/transcribe" {
            // Find JSON body after empty line
            if let bodyRange = request.range(of: "\r\n\r\n") {
                let body = String(request[bodyRange.upperBound...])
                handleTranscribe(client, body: body)
            } else {
                respond(client, status: 400, body: "{\"error\":\"no body\"}")
            }
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

        let segments = transcribe(filePath: req.path, language: lang)
        let response = TranscribeResponse(segments: segments)

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        if let jsonData = try? encoder.encode(response),
           let jsonStr = String(data: jsonData, encoding: .utf8) {
            respond(client, status: 200, body: jsonStr)
        } else {
            respond(client, status: 500, body: "{\"error\":\"encoding failed\"}")
        }
    }

    func respond(_ client: Int32, status: Int, body: String) {
        let statusText = status == 200 ? "OK" : "Error"
        let response = "HTTP/1.1 \(status) \(statusText)\r\nContent-Type: application/json\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)"
        write(client, response, response.utf8.count)
        close(client)
    }
}

// MARK: - Main

let args = CommandLine.arguments

if args.contains("--server") {
    let portIdx = args.firstIndex(of: "--port").map { args.index(after: $0) }
    let port = portIdx.flatMap { UInt16(args[$0]) } ?? 5300

    SFSpeechRecognizer.requestAuthorization { status in
        switch status {
        case .authorized:
            let server = HTTPServer(port: port)
            server.start()
        default:
            fputs("Error: Speech recognition not authorized (status: \(status.rawValue))\n", stderr)
            exit(1)
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
            let segments = transcribe(filePath: input, language: language)
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            if let data = try? encoder.encode(segments) {
                print(String(data: data, encoding: .utf8) ?? "[]")
            }
            exit(0)
        default:
            fputs("Error: Speech recognition not authorized (status: \(status.rawValue))\n", stderr)
            exit(1)
        }
    }
    RunLoop.main.run(until: Date(timeIntervalSinceNow: 300))
}
