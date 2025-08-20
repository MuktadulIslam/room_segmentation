// app/page.tsx
'use client'

import { useState, useRef, Suspense } from 'react'
import Image from 'next/image'
import dynamic from 'next/dynamic'
import Floor3DControls from '@/components/3d/Floor3DControls'

// Dynamically import 3D component to avoid SSR issues
const FloorPlane3D = dynamic(() => import('@/components/3d/FloorPlane3D'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-96 bg-gray-900 rounded-lg flex items-center justify-center">
      <div className="text-white">Loading 3D View...</div>
    </div>
  )
})

interface ApiResponse {
  success: boolean
  result_base64: string
  message: string
  edges?: {
    contours: Array<{
      contour_id: number
      points: Array<{ x: number; y: number }>
      area: number
    }>
    gradient_edges: Array<{
      x: number
      y: number
      magnitude: number
    }>
    boundary_points: Array<{ x: number; y: number }>
  }
}

type ProcessingType = 'remove-floor' | 'remove-wall'

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [processingType, setProcessingType] = useState<ProcessingType>('remove-floor')
  const [edgeData, setEdgeData] = useState<any>(null)
  const [imageOriginalSize, setImageOriginalSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 })
  
  // 3D Controls state
  const [show3D, setShow3D] = useState(false)
  const [showWireframe, setShowWireframe] = useState(false)
  
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        setError('Please select a valid image file')
        return
      }

      // Validate file size (10MB limit)
      if (file.size > 10 * 1024 * 1024) {
        setError('File size should be less than 10MB')
        return
      }

      setSelectedFile(file)
      setError(null)
      setProcessedImage(null)
      setEdgeData(null)
      setShow3D(false)

      // Create preview URL and get image dimensions
      const reader = new FileReader()
      reader.onload = (e) => {
        const img = document.createElement('img')
        img.onload = () => {
          setImageOriginalSize({ width: img.width, height: img.height })
        }
        img.src = e.target?.result as string
        setSelectedImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleFloorRemoval = async () => {
    if (!selectedFile) {
      setError('Please select an image first')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch('http://127.0.0.1:8000/remove-floor-base64', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: ApiResponse = await response.json()
      console.log('API Response:', data)
      
      if (data.success) {
        setProcessedImage(data.result_base64)
        if (data.edges) {
          setEdgeData(data.edges)
          console.log('Edge data received:', data.edges)
        }
      } else {
        throw new Error(data.message || 'Floor removal failed')
      }
    } catch (err) {
      console.error('Error:', err)
      setError(err instanceof Error ? err.message : 'An unexpected error occurred')
    } finally {
      setIsLoading(false)
    }
  }

  const handleWallRemoval = async () => {
    if (!selectedFile) {
      setError('Please select an image first')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      const response = await fetch('http://127.0.0.1:8000/remove-wall-base64', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: ApiResponse = await response.json()
      
      if (data.success) {
        setProcessedImage(data.result_base64)
        // Note: Wall removal doesn't provide edge data currently
        setEdgeData(null)
      } else {
        throw new Error(data.message || 'Wall removal failed')
      }
    } catch (err) {
      console.error('Error:', err)
      setError(err instanceof Error ? err.message : 'An unexpected error occurred')
    } finally {
      setIsLoading(false)
    }
  }

  const handleProcessing = () => {
    if (processingType === 'remove-floor') {
      handleFloorRemoval()
    } else {
      handleWallRemoval()
    }
  }

  const resetImages = () => {
    setSelectedImage(null)
    setProcessedImage(null)
    setSelectedFile(null)
    setError(null)
    setEdgeData(null)
    setShow3D(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const downloadImage = (imageData: string, filename: string) => {
    const link = document.createElement('a')
    link.href = imageData
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const getProcessingLabel = () => {
    switch (processingType) {
      case 'remove-floor':
        return 'Remove Floor'
      case 'remove-wall':
        return 'Remove Wall'
      default:
        return 'Process'
    }
  }

  const getResultTitle = () => {
    if (processedImage) {
      return processingType === 'remove-floor' ? 'Room Without Floor' : 'Room Without Wall'
    }
    return 'Result will appear here'
  }

  const getDownloadFilename = () => {
    return processingType === 'remove-floor' ? 'room-no-floor.png' : 'room-no-wall.png'
  }

  const getLoadingMessage = () => {
    switch (processingType) {
      case 'remove-floor':
        return 'Removing floor...'
      case 'remove-wall':
        return 'Removing wall...'
      default:
        return 'Processing...'
    }
  }

  const getEmptyStateMessage = () => {
    switch (processingType) {
      case 'remove-floor':
        return 'Floor-free image will appear here'
      case 'remove-wall':
        return 'Wall-free image will appear here'
      default:
        return 'Result will appear here'
    }
  }

  const canShow3D = () => {
    return processedImage && edgeData && processingType === 'remove-floor'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8">
      <div className="container mx-auto px-4 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            🏠 Room Segmentation Tool
          </h1>
          <p className="text-gray-600 text-lg">
            Upload a room image to automatically remove floors or walls with 3D visualization
          </p>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <div className="flex flex-col items-center">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
              id="image-upload"
            />
            <label
              htmlFor="image-upload"
              className="cursor-pointer bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors duration-200 mb-4"
            >
              Choose Image
            </label>
            
            {selectedFile && (
              <p className="text-sm text-gray-600 mb-4">
                Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
              </p>
            )}

            {/* Processing Type Selection */}
            {selectedImage && (
              <div className="flex flex-col items-center mb-4">
                <div className="flex bg-gray-100 rounded-lg p-1 mb-4">
                  <button
                    onClick={() => setProcessingType('remove-floor')}
                    className={`px-4 py-2 rounded-md transition-colors duration-200 ${
                      processingType === 'remove-floor'
                        ? 'bg-white text-blue-600 shadow-sm font-medium'
                        : 'text-gray-600 hover:text-gray-800'
                    }`}
                  >
                    Remove Floor
                  </button>
                  <button
                    onClick={() => setProcessingType('remove-wall')}
                    className={`px-4 py-2 rounded-md transition-colors duration-200 ${
                      processingType === 'remove-wall'
                        ? 'bg-white text-blue-600 shadow-sm font-medium'
                        : 'text-gray-600 hover:text-gray-800'
                    }`}
                  >
                    Remove Wall
                  </button>
                </div>

                <button
                  onClick={handleProcessing}
                  disabled={isLoading}
                  className={`font-semibold py-3 px-8 rounded-lg transition-colors duration-200 ${
                    isLoading
                      ? 'bg-gray-400 cursor-not-allowed'
                      : 'bg-green-500 hover:bg-green-600 text-white'
                  }`}
                >
                  {isLoading ? 'Processing...' : getProcessingLabel()}
                </button>
              </div>
            )}

            {(selectedImage || processedImage) && (
              <button
                onClick={resetImages}
                className="mt-4 bg-red-500 hover:bg-red-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors duration-200"
              >
                Reset
              </button>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-600 text-center">{error}</p>
            </div>
          )}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="mt-4 flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
              <span className="ml-3 text-gray-600">
                {getLoadingMessage()}
              </span>
            </div>
          )}
        </div>



        {/* Images Display */}
        {(selectedImage || processedImage) && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Original Image */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-semibold text-gray-800 mb-4 text-center">
                Original Image
              </h3>
              {selectedImage && (
                <div className="relative w-full h-80 rounded-lg overflow-hidden bg-gray-100">
                  <Image
                    src={selectedImage}
                    alt="Original room image"
                    fill
                    className="object-contain"
                  />
                </div>
              )}
            </div>

            {/* Processed Image */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-gray-800">
                  {getResultTitle()}
                </h3>
                <div className="flex space-x-2">
                  {processedImage && (
                    <button
                      onClick={() => downloadImage(
                        processedImage,
                        getDownloadFilename()
                      )}
                      className="bg-blue-500 hover:bg-blue-600 text-white text-sm px-3 py-1 rounded transition-colors duration-200"
                    >
                      Download
                    </button>
                  )}
                  {canShow3D() && (
                    <button
                      onClick={() => setShow3D(!show3D)}
                      className="bg-purple-500 hover:bg-purple-600 text-white text-sm px-3 py-1 rounded transition-colors duration-200"
                    >
                      {show3D ? 'Hide 3D' : 'Show 3D'}
                    </button>
                  )}
                </div>
              </div>
              
              {processedImage ? (
                <div className="relative w-full h-80 rounded-lg overflow-hidden bg-gray-100">
                  <div 
                    className="w-full h-full bg-transparent bg-opacity-50"
                    style={{
                      backgroundImage: `url("data:image/svg+xml,%3csvg width='100' height='100' xmlns='http://www.w3.org/2000/svg'%3e%3cdefs%3e%3cpattern id='pattern' width='20' height='20' patternUnits='userSpaceOnUse'%3e%3crect width='10' height='10' fill='%23f0f0f0'/%3e%3crect x='10' y='10' width='10' height='10' fill='%23f0f0f0'/%3e%3crect x='10' y='0' width='10' height='10' fill='%23e0e0e0'/%3e%3crect x='0' y='10' width='10' height='10' fill='%23e0e0e0'/%3e%3c/pattern%3e%3c/defs%3e%3crect width='100' height='100' fill='url(%23pattern)'/%3e%3c/svg%3e")`,
                    }}
                  >
                    <Image
                      src={processedImage}
                      alt={processingType === 'remove-floor' ? 'Room without floor' : 'Room without wall'}
                      fill
                      className="object-contain"
                    />
                  </div>
                </div>
              ) : (
                <div className="w-full h-80 rounded-lg bg-gray-50 border-2 border-dashed border-gray-300 flex items-center justify-center">
                  <p className="text-gray-500">
                    {getEmptyStateMessage()}
                  </p>
                </div>
              )}

              {/* Edge Data Info */}
              {edgeData && processingType === 'remove-floor' && (
                <div className="mt-4 p-3 bg-green-50 rounded-md">
                  <p className="text-sm text-green-700">
                    <strong>Edge Detection:</strong> Found {edgeData.contours?.length || 0} contours, {' '}
                    {edgeData.boundary_points?.length || 0} boundary points, and {' '}
                    {edgeData.gradient_edges?.length || 0} gradient edges.
                    {canShow3D() && (
                      <span className="ml-2 font-medium">
                        3D visualization available below!
                      </span>
                    )}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 3D Visualization Section - Separate from Images */}
        {canShow3D() && (
          <div className="mt-8">
            {/* 3D Controls */}
            <Floor3DControls
              show3D={show3D}
              setShow3D={setShow3D}
              showWireframe={showWireframe}
              setShowWireframe={setShowWireframe}
            />

            {/* 3D Visualization */}
            {show3D && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-xl font-semibold text-gray-800 mb-4 text-center">
                  🎯 3D Floor Plane Visualization
                </h3>
                <Suspense fallback={
                  <div className="w-full h-96 bg-gray-900 rounded-lg flex items-center justify-center">
                    <div className="text-white">Loading 3D Scene...</div>
                  </div>
                }>
                  <FloorPlane3D
                    edges={edgeData}
                    imageWidth={imageOriginalSize.width}
                    imageHeight={imageOriginalSize.height}
                    showWireframe={showWireframe}
                  />
                </Suspense>
                
                {/* 3D Instructions */}
                <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
                  <div className="flex items-center space-x-2">
                    <span className="text-blue-500 font-bold">🖱️</span>
                    <span>Click and drag to rotate the floor plane</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-green-500 font-bold">🔍</span>
                    <span>Scroll wheel to zoom in and out</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Instructions */}
        <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">
            How to Use:
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm text-gray-600">
            <div className="flex items-start">
              <span className="bg-blue-100 text-blue-600 rounded-full w-6 h-6 flex items-center justify-center font-semibold mr-3 mt-1">
                1
              </span>
              <p>Upload a room image with clearly visible floors, walls, and other elements</p>
            </div>
            <div className="flex items-start">
              <span className="bg-green-100 text-green-600 rounded-full w-6 h-6 flex items-center justify-center font-semibold mr-3 mt-1">
                2
              </span>
              <p>Choose between "Remove Floor" or "Remove Wall" mode</p>
            </div>
            <div className="flex items-start">
              <span className="bg-purple-100 text-purple-600 rounded-full w-6 h-6 flex items-center justify-center font-semibold mr-3 mt-1">
                3
              </span>
              <p>Click the process button to apply AI segmentation</p>
            </div>
            <div className="flex items-start">
              <span className="bg-orange-100 text-orange-600 rounded-full w-6 h-6 flex items-center justify-center font-semibold mr-3 mt-1">
                4
              </span>
              <p>View results, download the processed image, or explore in 3D</p>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <h4 className="font-semibold text-blue-800 mb-2">Processing Modes:</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-700">
              <div>
                <strong>Remove Floor:</strong> Creates a transparent PNG where the floor has been completely removed, showing only walls, furniture, and other room elements. Includes 3D visualization of the detected floor plane.
              </div>
              <div>
                <strong>Remove Wall:</strong> Creates a transparent PNG where the main wall has been removed, showing only floors, furniture, and other non-wall elements.
              </div>
            </div>
          </div>

          <div className="mt-4 p-4 bg-purple-50 rounded-lg">
            <h4 className="font-semibold text-purple-800 mb-2">3D Visualization Features:</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-purple-700">
              <div>
                <strong>Interactive 3D View:</strong> Visualize the detected floor plane as a 3D mesh with mouse controls for rotation, zoom, and pan.
              </div>
              <div>
                <strong>Edge Detection:</strong> Red points show detected boundary edges, with options to toggle wireframe mode and adjust plane thickness.
              </div>
            </div>
          </div>
          
          <div className="mt-4 p-4 bg-yellow-50 rounded-lg">
            <h4 className="font-semibold text-yellow-800 mb-2">Tips for Best Results:</h4>
            <ul className="text-sm text-yellow-700 space-y-1">
              <li>• Use high-quality images with good lighting</li>
              <li>• Ensure walls and floors are clearly distinguishable</li>
              <li>• Images should be at least 100x100 pixels</li>
              <li>• Maximum file size: 10MB</li>
              <li>• Supported formats: JPG, PNG, WebP</li>
              <li>• For best 3D results, use images with clear floor boundaries</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}