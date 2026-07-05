import Foundation
import Speech

struct Segment: Codable {
    let start: Double
    let end: Double
    let text: String
}

func parseArgs() -> (input: String, language: String) {
    let args = CommandLine.arguments
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
    return (input, language)
}

func transcribe(filePath: String, language: String) {
    let semaphore = DispatchSemaphore(value: 0)
    let url = URL(fileURLWithPath: filePath)
    let locale = Locale(identifier: language)

    guard let recognizer = SFSpeechRecognizer(locale: locale) else {
        fputs("Error: Speech recognizer not available for locale \(language)\n", stderr)
        exit(1)
    }

    guard recognizer.isAvailable else {
        fputs("Error: Speech recognizer is not available\n", stderr)
        exit(1)
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

    // Merge short segments into sentence-level chunks
    let merged = mergeSegments(allSegments, maxGap: 1.0, maxDuration: 10.0)

    let encoder = JSONEncoder()
    encoder.outputFormatting = .prettyPrinted
    if let data = try? encoder.encode(merged) {
        print(String(data: data, encoding: .utf8) ?? "[]")
    }
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

let (input, language) = parseArgs()

guard !input.isEmpty else {
    fputs("Usage: genie-speech-cli --input <file> [--language <locale>]\n", stderr)
    exit(1)
}

guard FileManager.default.fileExists(atPath: input) else {
    fputs("Error: File not found: \(input)\n", stderr)
    exit(1)
}

SFSpeechRecognizer.requestAuthorization { status in
    switch status {
    case .authorized:
        transcribe(filePath: input, language: language)
    default:
        fputs("Error: Speech recognition not authorized (status: \(status.rawValue))\n", stderr)
        exit(1)
    }
}

RunLoop.main.run(until: Date(timeIntervalSinceNow: 300))
