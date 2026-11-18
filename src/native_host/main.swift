import Foundation
import AppKit

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate

// Signal handlers for graceful exit
signal(SIGINT) { _ in
    print("\nReceived SIGINT. Shutting down EyeMouse...")
    NSApp.terminate(nil)
}
signal(SIGTERM) { _ in
    print("\nReceived SIGTERM. Shutting down EyeMouse...")
    NSApp.terminate(nil)
}

app.run()

