import Foundation
import AppKit

public final class MenuBarController: NSObject, NSMenuDelegate {
    public static let shared = MenuBarController()
    
    private var statusItem: NSStatusItem!
    private var menu: NSMenu!
    
    private var statusMenuItem: NSMenuItem!
    private var cameraMenu: NSMenu!
    private var cameraMenuItem: NSMenuItem!
    private var shortcutMenuItem: NSMenuItem!
    
    private override init() {
        super.init()
    }
    
    public func setupMenuBar() {
        self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        
        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "eye", accessibilityDescription: "EyeMouse")
            button.imagePosition = .imageLeft
            button.title = " EyeMouse"
        }
        
        self.menu = NSMenu()
        menu.delegate = self
        
        // 1. Status header (disabled item)
        self.statusMenuItem = NSMenuItem(title: "EyeMouse: Idle (Hold shortcut to activate)", action: nil, keyEquivalent: "")
        statusMenuItem.isEnabled = false
        menu.addItem(statusMenuItem)
        
        menu.addItem(NSMenuItem.separator())
        
        // 2. Camera submenu
        self.cameraMenu = NSMenu(title: "Select Camera")
        self.cameraMenuItem = NSMenuItem(title: "Selected Camera: Loading...", action: nil, keyEquivalent: "")
        cameraMenuItem.submenu = cameraMenu
        menu.addItem(cameraMenuItem)
        
        // 3. Current Activation Shortcut
        let currentShortcut = ShortcutConfig.load().displayString
        self.shortcutMenuItem = NSMenuItem(title: "Activation Shortcut: \(currentShortcut) (Hold)", action: nil, keyEquivalent: "")
        shortcutMenuItem.isEnabled = false
        menu.addItem(shortcutMenuItem)
        
        menu.addItem(NSMenuItem.separator())
        
        // 4. Calibration
        let calibMenu = NSMenu(title: "Calibration")
        let calibMainItem = NSMenuItem(title: "Gaze Calibration", action: nil, keyEquivalent: "")
        calibMainItem.submenu = calibMenu
        
        let fivePointItem = NSMenuItem(title: "Interactive Calibration (5 Points)...", action: #selector(openFivePointCalibration), keyEquivalent: "c")
        fivePointItem.target = self
        calibMenu.addItem(fivePointItem)
        
        let onePointItem = NSMenuItem(title: "Quick 1-Point Recalibrate (Press Space)...", action: #selector(openOnePointCalibration), keyEquivalent: "1")
        onePointItem.target = self
        calibMenu.addItem(onePointItem)
        
        let ninePointItem = NSMenuItem(title: "Interactive Calibration (9 Points)...", action: #selector(openNinePointCalibration), keyEquivalent: "9")
        ninePointItem.target = self
        calibMenu.addItem(ninePointItem)
        
        calibMenu.addItem(NSMenuItem.separator())
        
        let resetPriorItem = NSMenuItem(title: "Reset Model to Base Prior", action: #selector(resetToBasePrior), keyEquivalent: "")
        resetPriorItem.target = self
        calibMenu.addItem(resetPriorItem)
        
        let batchWizardItem = NSMenuItem(title: "Batch 13-Point Validation Wizard...", action: #selector(openBatchCalibration), keyEquivalent: "")
        batchWizardItem.target = self
        calibMenu.addItem(batchWizardItem)
        
        menu.addItem(calibMainItem)
        
        // 5. Diagnostic Gaze Marker Overlay
        let previewItem = NSMenuItem(title: "Toggle Diagnostic Gaze Marker", action: #selector(toggleGazePreview), keyEquivalent: "m")
        previewItem.target = self
        menu.addItem(previewItem)
        
        menu.addItem(NSMenuItem.separator())
        
        // 6. Tracking Diagnostics
        let diagItem = NSMenuItem(title: "Tracking Diagnostics...", action: #selector(openDiagnostics), keyEquivalent: "d")
        diagItem.target = self
        menu.addItem(diagItem)
        
        // 7. Settings
        let settingsItem = NSMenuItem(title: "Settings...", action: #selector(openSettings), keyEquivalent: ",")
        settingsItem.target = self
        menu.addItem(settingsItem)
        
        menu.addItem(NSMenuItem.separator())
        
        // 8. Quit EyeMouse
        let quitItem = NSMenuItem(title: "Quit EyeMouse", action: #selector(quitApp), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)
        
        statusItem.menu = menu
        
        // Observe State Changes
        SafetyStateManager.shared.onStateChanged = { [weak self] state in
            self?.updateStatus(for: state)
        }
        
        CameraManager.shared.onSelectedDeviceChanged = { [weak self] dev in
            self?.updateCameraMenu()
        }
        CameraManager.shared.onDevicesChanged = { [weak self] _ in
            self?.updateCameraMenu()
        }
        
        updateStatus(for: SafetyStateManager.shared.currentState)
        updateCameraMenu()
    }
    
    public func updateStatus(for state: SafetyState) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.statusMenuItem.title = "EyeMouse: \(state.displayTitle)"
            
            if let button = self.statusItem.button {
                switch state {
                case .inactive:
                    button.image = NSImage(systemSymbolName: "eye", accessibilityDescription: "EyeMouse Idle")
                    button.title = " EyeMouse"
                case .starting:
                    button.image = NSImage(systemSymbolName: "eye.circle", accessibilityDescription: "EyeMouse Starting")
                    button.title = " Starting..."
                case .active:
                    button.image = NSImage(systemSymbolName: "eye.fill", accessibilityDescription: "EyeMouse Active")
                    button.title = " [Active]"
                case .stopping:
                    button.image = NSImage(systemSymbolName: "eye", accessibilityDescription: "EyeMouse Stopping")
                    button.title = " Stopping..."
                case .error:
                    button.image = NSImage(systemSymbolName: "exclamationmark.triangle.fill", accessibilityDescription: "EyeMouse Error")
                    button.title = " Error"
                }
            }
        }
    }
    
    public func updateCameraMenu() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            let selName = CameraManager.shared.selectedDevice?.localizedName ?? "None"
            self.cameraMenuItem.title = "Camera: \(selName)"
            
            self.cameraMenu.removeAllItems()
            for dev in CameraManager.shared.availableDevices {
                let item = NSMenuItem(
                    title: "\(dev.localizedName) (\(dev.type.rawValue))",
                    action: #selector(self.selectCameraDevice(_:)),
                    keyEquivalent: ""
                )
                item.target = self
                item.representedObject = dev.id
                if dev.id == CameraManager.shared.selectedDevice?.id {
                    item.state = .on
                }
                self.cameraMenu.addItem(item)
            }
            if CameraManager.shared.availableDevices.isEmpty {
                let item = NSMenuItem(title: "No Cameras Found", action: nil, keyEquivalent: "")
                item.isEnabled = false
                self.cameraMenu.addItem(item)
            }
        }
    }
    
    public func updateShortcutDisplay() {
        let sc = ShortcutConfig.load()
        shortcutMenuItem.title = "Activation Shortcut: \(sc.displayString) (Hold)"
    }
    
    @objc private func selectCameraDevice(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String else { return }
        CameraManager.shared.selectDevice(byID: id)
    }
    
    @objc private func openFivePointCalibration() {
        CalibrationWindowController.shared.show()
        CalibrationWindowController.shared.startAdaptiveCalibration(pointCount: 5)
        NSApp.activate(ignoringOtherApps: true)
    }
    
    @objc private func openOnePointCalibration() {
        CalibrationWindowController.shared.show()
        CalibrationWindowController.shared.startQuickOnePoint()
        NSApp.activate(ignoringOtherApps: true)
    }
    
    @objc private func openNinePointCalibration() {
        CalibrationWindowController.shared.show()
        CalibrationWindowController.shared.startAdaptiveCalibration(pointCount: 9)
        NSApp.activate(ignoringOtherApps: true)
    }
    
    @objc private func resetToBasePrior() {
        CalibrationWindowController.shared.resetAdaptationToBasePrior()
    }
    
    @objc private func openBatchCalibration() {
        CalibrationWindowController.shared.show()
        CalibrationWindowController.shared.startBatchCalibration()
        NSApp.activate(ignoringOtherApps: true)
    }
    
    @objc private func openCalibration() {
        CalibrationWindowController.shared.show()
        NSApp.activate(ignoringOtherApps: true)
    }
    
    @objc private func toggleGazePreview() {
        LiveGazePreviewController.shared.toggle()
    }
    
    @objc private func openDiagnostics() {
        DiagnosticsWindowController.shared.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
    
    @objc private func openSettings() {
        SettingsWindowController.shared.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
    
    @objc private func quitApp() {
        NSApplication.shared.terminate(nil)
    }
}

