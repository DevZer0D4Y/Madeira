// SPDX-License-Identifier: GPL-3.0-or-later
// Madeira Converter Exception: see LICENSE-EXCEPTION.md

import Foundation
import SwiftUI

/// No JIT (experimental), under Settings › JIT: Madeira asks for no debugger at all
/// (no StikDebug, no pairing file, no LocalDevVPN) and runs x86-64 code with its own
/// interpreter (X64Interpreter.c) instead of FEX's JIT. See docs/NO-JIT.md.
///
/// So far only the engine check runs that way. Wine's own Windows libraries are loaded
/// into the JIT pool too, so a game cannot start in this mode yet; Play says so instead
/// of asking for JIT, and choosing another method under Settings › JIT plays as before.
@MainActor
final class NoJITEngine: ObservableObject {
    static let shared = NoJITEngine()

    enum State: Equatable {
        case idle
        case running
        case passed
        case failed
    }

    @Published private(set) var state: State = .idle
    /// "24 of 24 checks matched…", from the last check.
    @Published private(set) var summary: String?
    /// One line per program.
    @Published private(set) var details = ""
    /// Whether the last check ran with CS_DEBUGGED clear (no JIT): nil when the kernel
    /// did not say.
    @Published private(set) var ranWithoutJIT: Bool?

    static var selected: Bool { JITCoordinator.shared.method == .noJIT }

    static let playMessage = L("No JIT can't start games yet: Madeira's x86-64 interpreter works, but Wine's own Windows libraries still need JIT. To play now, choose Automatic under Settings › JIT.")
    static let enableMessage = L("No JIT is selected under Settings › JIT, so Madeira doesn't ask for a debugger. To enable JIT, choose Automatic, StikDebug or Built-in StikJIT.")

    func run() {
        guard state != .running else { return }
        state = .running
        LogStore.shared.log("[no-jit] engine check started")
        Task.detached(priority: .userInitiated) {
            var report = MadeiraNoJITReport()
            let size = 4096
            var text = [CChar](repeating: 0, count: size)
            let passed = madeira_nojit_self_check(&report, &text, size)
            let lines = text.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
                .split(separator: "\n").map(String.init)
            var flags: UInt32 = 0
            let known = jit_cs_status(&flags)
            let withoutJIT: Bool? = known ? (flags & 0x1000_0000) == 0 : nil   // CS_DEBUGGED
            await MainActor.run {
                let engine = NoJITEngine.shared
                engine.state = passed ? .passed : .failed
                engine.summary = lines.last
                engine.details = lines.dropLast().joined(separator: "\n")
                engine.ranWithoutJIT = withoutJIT
                LogStore.shared.log("[no-jit] engine check \(passed ? "passed" : "FAILED"): \(lines.last ?? "")"
                                    + " (CS_DEBUGGED \(withoutJIT.map { $0 ? "clear" : "set" } ?? "unknown"))",
                                    level: passed ? .success : .error)
                for line in lines.dropLast() where line.hasPrefix("FAIL") {
                    LogStore.shared.log("[no-jit] \(line)", level: .error)
                }
            }
        }
    }
}

/// The No JIT part of JIT setup: what the mode is, and the engine check.
struct NoJITEngineSection: View {
    @ObservedObject private var engine = NoJITEngine.shared

    var body: some View {
        Section {
            Button {
                engine.run()
            } label: {
                if engine.state == .running {
                    HStack { ProgressView(); Text("Checking the engine…") }
                } else {
                    Label("Run engine check", systemImage: "cpu")
                }
            }
            .disabled(engine.state == .running)
            if let summary = engine.summary, engine.state != .running {
                Label(summary, systemImage: engine.state == .passed ? "checkmark.circle.fill" : "xmark.octagon.fill")
                    .foregroundStyle(engine.state == .passed ? Color.green : Color.red)
            }
            if let withoutJIT = engine.ranWithoutJIT, engine.state != .running {
                LabeledContent("JIT during the check", value: withoutJIT ? L("Off") : L("On"))
            }
            if !engine.details.isEmpty, engine.state != .running {
                DisclosureGroup("Details") {
                    Text(engine.details)
                        .font(.caption2.monospaced())
                        .textSelection(.enabled)
                }
            }
        } header: {
            Text("No JIT engine")
        } footer: {
            Text("Madeira's x86 interpreter runs Windows code, 64-bit and 32-bit with x87 and MMX, without executable memory, so it needs no debugger, StikDebug or LocalDevVPN. The engine check runs small x86 programs through it and compares every answer with a real x86 CPU's. Games can't start this way yet: Wine's own libraries still need JIT, and interpreted code is much slower than JIT.")
        }
    }
}
