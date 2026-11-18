import Foundation
import AppKit
import SwiftUI

public final class LiveGazePreviewController: ObservableObject {
    public static let shared = LiveGazePreviewController()
    
    private var window: NSWindow?
    
    private init() {}
    
    public func toggle() {
        if window == nil {
            show()
        } else {
            hide()
        }
    }
    
    public func show() {
        guard window == nil else { return }
        guard let screen = NSScreen.main else { return }
        
        let overlayWindow = NSWindow(
            contentRect: screen.frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false,
            screen: screen
        )
        overlayWindow.isOpaque = false
        overlayWindow.backgroundColor = .clear
        overlayWindow.level = .floating
        overlayWindow.ignoresMouseEvents = true
        overlayWindow.collectionBehavior = [.canJoinAllSpaces, .stationary]
        
        let rootView = LiveGazeOverlayView()
        overlayWindow.contentView = NSHostingView(rootView: rootView)
        overlayWindow.orderFrontRegardless()
        self.window = overlayWindow
        
        DispatchQueue.main.async {
            DiagnosticsStore.shared.showLiveGazePreview = true
        }
    }
    
    public func hide() {
        window?.orderOut(nil)
        window = nil
        DispatchQueue.main.async {
            DiagnosticsStore.shared.showLiveGazePreview = false
        }
    }
}

struct LiveGazeOverlayView: View {
    @ObservedObject var diagnostics = DiagnosticsStore.shared
    
    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Dim border indicator to show diagnostic overlay is active
                Rectangle()
                    .stroke(Color.cyan.opacity(0.3), lineWidth: 2)
                    .allowsHitTesting(false)
                
                // Top banner
                VStack {
                    HStack(spacing: 12) {
                        Circle()
                            .fill(diagnostics.faceDetected ? Color.green : Color.red)
                            .frame(width: 8, height: 8)
                        
                        Text("DIAGNOSTIC GAZE MARKER — NO CURSOR INJECTION")
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundColor(.white)
                        
                        Text("(\(diagnostics.activeProfileName) | \(diagnostics.activeQualityGrade))")
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundColor(.cyan)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 6)
                    .background(Color.black.opacity(0.75))
                    .cornerRadius(16)
                    .padding(.top, 40)
                    
                    Spacer()
                }
                
                // Gaze Marker
                if diagnostics.faceDetected && diagnostics.isGazeValid {
                    let point = diagnostics.screenPoint
                    let clampedX = min(max(point.x, 20.0), geo.size.width - 20.0)
                    let clampedY = min(max(point.y, 20.0), geo.size.height - 20.0)
                    
                    ZStack {
                        // Outer confidence ring
                        Circle()
                            .stroke(Color.cyan.opacity(0.5 * diagnostics.confidence), lineWidth: 2)
                            .frame(width: 48, height: 48)
                        
                        // Inner reticle
                        Circle()
                            .fill(Color.cyan.opacity(0.85))
                            .frame(width: 16, height: 16)
                            .shadow(color: .cyan, radius: 8)
                        
                        // Center pip
                        Circle()
                            .fill(Color.white)
                            .frame(width: 4, height: 4)
                        
                        // Crosshair ticks
                        Path { path in
                            path.move(to: CGPoint(x: 0, y: -14))
                            path.addLine(to: CGPoint(x: 0, y: -6))
                            path.move(to: CGPoint(x: 0, y: 6))
                            path.addLine(to: CGPoint(x: 0, y: 14))
                            path.move(to: CGPoint(x: -14, y: 0))
                            path.addLine(to: CGPoint(x: -6, y: 0))
                            path.move(to: CGPoint(x: 6, y: 0))
                            path.addLine(to: CGPoint(x: 14, y: 0))
                        }
                        .stroke(Color.white.opacity(0.9), lineWidth: 1.5)
                    }
                    .position(x: clampedX, y: clampedY)
                    .animation(.interactiveSpring(response: 0.15, dampingFraction: 0.8), value: point)
                }
            }
        }
        .edgesIgnoringSafeArea(.all)
    }
}

