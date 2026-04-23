import Foundation
import Metal

enum EnvProjectError: Error, CustomStringConvertible {
    case noMetalDevice
    case shaderFileMissing(String)
    case shaderCompilationFailed(String)
    case functionMissing(String)
    case pipelineCreationFailed(String)
    case commandQueueUnavailable
    case commandBufferUnavailable
    case encoderUnavailable
    case bufferAllocationFailed(String)
    case unexpectedStructLayout(name: String, size: Int, stride: Int, alignment: Int)
    case validationFailed(message: String)

    var description: String {
        switch self {
        case .noMetalDevice:
            return "No Metal device is available."
        case let .shaderFileMissing(path):
            return "Shader file not found at \(path)."
        case let .shaderCompilationFailed(message):
            return "Failed to compile Metal shader: \(message)"
        case let .functionMissing(name):
            return "Compiled Metal library is missing function \(name)."
        case let .pipelineCreationFailed(message):
            return "Failed to create compute pipeline or execute commands: \(message)"
        case .commandQueueUnavailable:
            return "Failed to create command queue."
        case .commandBufferUnavailable:
            return "Failed to create command buffer."
        case .encoderUnavailable:
            return "Failed to create compute command encoder."
        case let .bufferAllocationFailed(name):
            return "Failed to allocate \(name) buffer."
        case let .unexpectedStructLayout(name, size, stride, alignment):
            return "Unexpected \(name) layout: size=\(size) stride=\(stride) alignment=\(alignment)."
        case let .validationFailed(message):
            return message
        }
    }
}

func loadShaderSource(from path: String) throws -> String {
    guard FileManager.default.fileExists(atPath: path) else {
        throw EnvProjectError.shaderFileMissing(path)
    }

    return try String(contentsOfFile: path, encoding: .utf8)
}

func checkLayout<T>(
    _ type: T.Type,
    name: String,
    expectedSize: Int,
    expectedStride: Int,
    expectedAlignment: Int
) throws {
    let actualSize = MemoryLayout<T>.size
    let actualStride = MemoryLayout<T>.stride
    let actualAlignment = MemoryLayout<T>.alignment

    if actualSize != expectedSize || actualStride != expectedStride || actualAlignment != expectedAlignment {
        throw EnvProjectError.unexpectedStructLayout(
            name: name,
            size: actualSize,
            stride: actualStride,
            alignment: actualAlignment
        )
    }
}

func makeLibrary(device: MTLDevice, shaderPath: String) throws -> MTLLibrary {
    let shaderSource = try loadShaderSource(from: shaderPath)

    do {
        return try device.makeLibrary(source: shaderSource, options: nil)
    } catch {
        throw EnvProjectError.shaderCompilationFailed(String(describing: error))
    }
}

func makePipeline(device: MTLDevice, library: MTLLibrary, name: String) throws -> MTLComputePipelineState {
    guard let function = library.makeFunction(name: name) else {
        throw EnvProjectError.functionMissing(name)
    }

    do {
        return try device.makeComputePipelineState(function: function)
    } catch {
        throw EnvProjectError.pipelineCreationFailed(String(describing: error))
    }
}

func copyArray<T>(_ values: [T], to buffer: MTLBuffer) {
    let pointer = buffer.contents().bindMemory(to: T.self, capacity: values.count)
    for index in values.indices {
        pointer[index] = values[index]
    }
}

func readArray<T>(from buffer: MTLBuffer, count: Int) -> [T] {
    let pointer = buffer.contents().bindMemory(to: T.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}

func writeValue<T>(_ value: inout T, to buffer: MTLBuffer) {
    withUnsafeBytes(of: &value) { bytes in
        if let baseAddress = bytes.baseAddress {
            buffer.contents().copyMemory(from: baseAddress, byteCount: bytes.count)
        }
    }
}

func runComputePass(
    commandQueue: MTLCommandQueue,
    pipeline: MTLComputePipelineState,
    count: Int,
    configure: (MTLComputeCommandEncoder) -> Void
) throws {
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
        throw EnvProjectError.commandBufferUnavailable
    }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw EnvProjectError.encoderUnavailable
    }

    encoder.setComputePipelineState(pipeline)
    configure(encoder)

    let threadsPerThreadgroup = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
    let threadsPerGrid = MTLSize(width: count, height: 1, depth: 1)
    encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    if let error = commandBuffer.error {
        throw EnvProjectError.pipelineCreationFailed(String(describing: error))
    }
}
