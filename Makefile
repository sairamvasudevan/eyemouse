# EyeMouse Root Makefile
# Provides reproducible compilation and testing for Swift and Python components.

SDK_PATH ?= /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
TARGET_ARCH ?= arm64-apple-macos14.0
SWIFTC ?= /usr/bin/swiftc
SWIFT_FLAGS = -sdk $(SDK_PATH) -target $(TARGET_ARCH) -O

VENV_PYTHON = .venv/bin/python
PYTEST = .venv/bin/pytest

.PHONY: all test-python test-swift build-swift-smoke build-swift-ipc test-ipc freeze-engine benchmark build-app test-m3-safety test-m3-live test-all clean

all: test-python build-swift-smoke build-swift-ipc build-app

test-python:
	$(PYTEST) tests/ -v

build-swift-smoke:
	mkdir -p build
	$(SWIFTC) $(SWIFT_FLAGS) -o build/swift_smoke tests/swift_smoke/main.swift

build-swift-ipc:
	mkdir -p build
	/usr/bin/clang -isysroot $(SDK_PATH) -target $(TARGET_ARCH) -O -c tests/swift_ipc_producer/shm_bridge.c -o build/shm_bridge.o
	$(SWIFTC) $(SWIFT_FLAGS) -o build/swift_ipc_producer tests/swift_ipc_producer/main.swift build/shm_bridge.o

test-swift: build-swift-smoke
	./build/swift_smoke

test-ipc: build-swift-ipc
	$(VENV_PYTHON) tests/test_swift_ipc_runner.py

freeze-engine:
	.venv/bin/pyinstaller packaging/eyemouse_engine.spec --noconfirm --distpath dist_engine --workpath build_engine

benchmark: build-swift-ipc
	$(VENV_PYTHON) tests/benchmarks.py

build-app:
	mkdir -p build/EyeMouse.app/Contents/MacOS build/EyeMouse.app/Contents/Resources/engine
	/usr/bin/clang -isysroot $(SDK_PATH) -target $(TARGET_ARCH) -O -c src/native_host/Bridge/shm_bridge.c -o build/shm_bridge.o
	$(SWIFTC) $(SWIFT_FLAGS) -import-objc-header src/native_host/Bridge/shm_bridge.h -o build/EyeMouse \
	  src/native_host/Config/*.swift \
	  src/native_host/Keyboard/*.swift \
	  src/native_host/Core/*.swift \
	  src/native_host/Camera/*.swift \
	  src/native_host/Calibration/*.swift \
	  src/native_host/IPC/*.swift \
	  src/native_host/UI/*.swift \
	  src/native_host/AppDelegate.swift \
	  src/native_host/main.swift \
	  build/shm_bridge.o
	cp build/EyeMouse build/EyeMouse.app/Contents/MacOS/
	cp packaging/Info.plist build/EyeMouse.app/Contents/
	cp packaging/AppIcon.icns build/EyeMouse.app/Contents/Resources/
	cp dist_engine/eyemouse_engine/eyemouse_engine build/EyeMouse.app/Contents/Resources/engine/
	cp -R dist_engine/eyemouse_engine/_internal build/EyeMouse.app/Contents/Resources/engine/
	codesign -s - --force --identifier "com.eyemouse.EyeMouse" build/EyeMouse.app/Contents/MacOS/EyeMouse
	codesign -s - --force --identifier "com.eyemouse.EyeMouse" build/EyeMouse.app

test-m3-safety:
	mkdir -p build
	$(SWIFTC) $(SWIFT_FLAGS) -o build/test_m3_safety_state \
	  src/native_host/Config/*.swift \
	  src/native_host/Keyboard/KeyCodeTable.swift \
	  src/native_host/Core/SafetyState.swift \
	  tests/test_m3_safety_state.swift
	./build/test_m3_safety_state

test-m3-live:
	mkdir -p build
	/usr/bin/clang -isysroot $(SDK_PATH) -target $(TARGET_ARCH) -O -c src/native_host/Bridge/shm_bridge.c -o build/shm_bridge.o
	$(SWIFTC) $(SWIFT_FLAGS) -import-objc-header src/native_host/Bridge/shm_bridge.h -o build/test_m3_live_pipeline \
	  src/native_host/Config/*.swift \
	  src/native_host/Keyboard/KeyCodeTable.swift \
	  src/native_host/Core/*.swift \
	  src/native_host/Camera/*.swift \
	  src/native_host/Calibration/*.swift \
	  src/native_host/IPC/*.swift \
	  tests/test_m3_live_pipeline.swift \
	  build/shm_bridge.o
	./build/test_m3_live_pipeline

test-m4-live:
	mkdir -p build
	/usr/bin/clang -isysroot $(SDK_PATH) -target $(TARGET_ARCH) -O -c src/native_host/Bridge/shm_bridge.c -o build/shm_bridge.o
	$(SWIFTC) $(SWIFT_FLAGS) -import-objc-header src/native_host/Bridge/shm_bridge.h -o build/test_m4_live_calibration \
	  src/native_host/Config/*.swift \
	  src/native_host/Keyboard/KeyCodeTable.swift \
	  src/native_host/Core/*.swift \
	  src/native_host/Camera/*.swift \
	  src/native_host/Calibration/*.swift \
	  src/native_host/IPC/*.swift \
	  tests/test_m4_live_calibration.swift \
	  build/shm_bridge.o
	./build/test_m4_live_calibration

test-all: test-python test-swift test-ipc benchmark test-m3-safety test-m3-live test-m4-live

clean:
	rm -rf build build_smoke dist_smoke build_engine dist_engine EyeMouse.app



