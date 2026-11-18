import Foundation
import AppKit
import SwiftUI

public final class DiagnosticsWindowController: NSWindowController {
    public static let shared = DiagnosticsWindowController()
    
    private convenience init() {
        let window = NSWindow(
            contentRect: NSRect(x: 100, y: 100, width: 620, height: 680),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "EyeMouse Tracking Diagnostics"
        window.center()
        window.contentViewController = NSHostingController(rootView: DiagnosticsView())
        self.init(window: window)
    }
}

public struct DiagnosticsView: View {
    @ObservedObject var store = DiagnosticsStore.shared
    
    public init() {}
    
    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                // 1. Milestone 4 Calibration & Safety Notice
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Image(systemName: store.isCalibrated ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                            .foregroundColor(store.isCalibrated ? .green : .orange)
                        Text(store.isCalibrated ? "PERSONALIZED CALIBRATION ACTIVE (MILESTONE 4)" : "UNCALIBRATED OPTICAL PROXY — CALIBRATION REQUIRED")
                            .font(.headline)
                            .foregroundColor(store.isCalibrated ? .green : .orange)
                    }
                    if store.isCalibrated {
                        Text("Model: \(store.activeProfileName) | Quality: \(store.activeQualityGrade) | Median Error: \(String(format: "%.1f", store.activeMedianErrorPixels)) px. NO OS CURSOR INJECTION: Gaze displayed via diagnostic marker only.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    } else {
                        Text("Gaze coordinates are uncalibrated optical proxies. Click 'Calibrate Eye Tracking...' to personalize screen gaze.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    
                    HStack(spacing: 12) {
                        Button("Calibrate Eye Tracking...") {
                            CalibrationWindowController.shared.show()
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        
                        Button(store.showLiveGazePreview ? "Hide Gaze Marker" : "Show Diagnostic Gaze Marker") {
                            LiveGazePreviewController.shared.toggle()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                    }
                    .padding(.top, 4)
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(store.isCalibrated ? Color.green.opacity(0.12) : Color.orange.opacity(0.12))
                .cornerRadius(8)
                
                // 2. Camera Status
                GroupBox(label: Label("Camera Acquisition", systemImage: "camera.fill")) {
                    HStack(spacing: 20) {
                        HStack(spacing: 8) {
                            Circle()
                                .fill(store.isCameraActive ? Color.green : Color.gray)
                                .frame(width: 12, height: 12)
                            Text(store.isCameraActive ? "Camera Active (Streaming)" : "Camera Idle (Dormant)")
                                .font(.subheadline)
                                .bold()
                        }
                        Spacer()
                        VStack(alignment: .trailing) {
                            Text("Capture: \(String(format: "%.1f", store.cameraFps)) FPS")
                                .font(.subheadline)
                                .monospacedDigit()
                            Text("Device: \(CameraManager.shared.selectedDevice?.localizedName ?? "None")")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }
                
                // 3. Face Detection & Presence
                GroupBox(label: Label("Face Presence", systemImage: "person.crop.circle.badge.checkmark")) {
                    HStack {
                        if store.faceDetected {
                            Label("FACE DETECTED", systemImage: "checkmark.circle.fill")
                                .font(.subheadline)
                                .foregroundColor(.green)
                                .bold()
                        } else {
                            Label("NO FACE DETECTED", systemImage: "xmark.circle.fill")
                                .font(.subheadline)
                                .foregroundColor(.red)
                                .bold()
                        }
                        Spacer()
                        Text("Confidence: \(String(format: "%.1f%%", store.confidence * 100))")
                            .font(.subheadline)
                            .monospacedDigit()
                    }
                    .padding(.vertical, 4)
                }
                
                // 4. Normalized Iris Features
                GroupBox(label: Label("Normalized Iris Position (Local Eye Coordinates)", systemImage: "eye.circle")) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Left Iris (Norm UV):")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("(\(String(format: "%+.3f", store.leftIrisX)), \(String(format: "%+.3f", store.leftIrisY)))")
                                .font(.subheadline)
                                .monospacedDigit()
                        }
                        HStack {
                            Text("Right Iris (Norm UV):")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("(\(String(format: "%+.3f", store.rightIrisX)), \(String(format: "%+.3f", store.rightIrisY)))")
                                .font(.subheadline)
                                .monospacedDigit()
                        }
                        Divider()
                        HStack {
                            Text(store.isCalibrated ? "Calibrated Screen Gaze (Norm):" : "Optical Gaze Proxy (Filtered):")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("(\(String(format: "%+.3f", store.gazeX)), \(String(format: "%+.3f", store.gazeY)))")
                                .font(.subheadline)
                                .bold()
                                .monospacedDigit()
                        }
                        if store.isCalibrated {
                            let geom = CalibrationGeometry.current
                            let retinaPt = geom.pointsToPixels(point: store.screenPoint)
                            HStack {
                                Text("Display Points (CG):")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Spacer()
                                Text("(\(String(format: "%.1f", store.screenPoint.x)), \(String(format: "%.1f", store.screenPoint.y))) pt")
                                    .font(.caption)
                                    .monospacedDigit()
                            }
                            HStack {
                                Text("Retina Resolution:")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Spacer()
                                Text("(\(String(format: "%.0f", retinaPt.x)), \(String(format: "%.0f", retinaPt.y))) px")
                                    .font(.caption)
                                    .monospacedDigit()
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }
                
                // 5. Blink Dynamics
                GroupBox(label: Label("Blink Dynamics (Hysteresis 0.55 / 0.35)", systemImage: "eyes")) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Blink Phase:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(store.blinkPhase)
                                .font(.subheadline)
                                .bold()
                                .foregroundColor(store.blinkPhase == "CLOSED" ? .yellow : .primary)
                        }
                        HStack {
                            Text("Left Score: \(String(format: "%.2f", store.leftBlinkScore)) | Right Score: \(String(format: "%.2f", store.rightBlinkScore))")
                                .font(.caption)
                                .monospacedDigit()
                            Spacer()
                            Text("Duration: \(String(format: "%.0f ms", store.blinkDurationMs))")
                                .font(.caption)
                                .monospacedDigit()
                        }
                        if store.isSustainedBlink {
                            HStack {
                                Image(systemName: "hand.tap.fill")
                                    .foregroundColor(.blue)
                                Text("Potential Intentional Blink (Duration >= 350ms)")
                                    .font(.caption)
                                    .foregroundColor(.blue)
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }
                
                // 6. 3D Head Pose
                GroupBox(label: Label("3D Head Pose (SolvePnP Euler Angles)", systemImage: "gyroscope")) {
                    HStack(spacing: 20) {
                        VStack {
                            Text("Yaw (Turn)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("\(String(format: "%+.1f°", store.headYaw))")
                                .font(.headline)
                                .monospacedDigit()
                        }
                        Spacer()
                        VStack {
                            Text("Pitch (Tilt)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("\(String(format: "%+.1f°", store.headPitch))")
                                .font(.headline)
                                .monospacedDigit()
                        }
                        Spacer()
                        VStack {
                            Text("Roll (Lean)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("\(String(format: "%+.1f°", store.headRoll))")
                                .font(.headline)
                                .monospacedDigit()
                        }
                    }
                    .padding(.vertical, 4)
                }
                
                // 7. Performance & IPC Telemetry
                GroupBox(label: Label("System Telemetry & Latency", systemImage: "speedometer")) {
                    VStack(spacing: 6) {
                        HStack {
                            Text("Vision Engine Rate:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("\(String(format: "%.1f", store.engineFps)) FPS")
                                .font(.subheadline)
                                .monospacedDigit()
                        }
                        HStack {
                            Text("Inference Latency:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("Mean: \(String(format: "%.1f ms", store.engineLatencyMs)) | p95: \(String(format: "%.1f ms", store.p95LatencyMs))")
                                .font(.subheadline)
                                .monospacedDigit()
                        }
                        HStack {
                            Text("Frames Published / Dropped:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("\(store.publishedFrames) / \(store.droppedFrames)")
                                .font(.subheadline)
                                .monospacedDigit()
                        }
                        HStack {
                            Text("Engine Process:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Spacer()
                            Text(store.engineStatus)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
            .padding(16)
        }
        .frame(minWidth: 540, minHeight: 640)
    }
}

