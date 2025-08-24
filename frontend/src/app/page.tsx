// app/page.tsx
'use client'

import { useState, useRef } from 'react'
import Image from 'next/image'

// Simple API response interface
interface ApiResponse {
  success: boolean
  result_base64: string
  message: string
}

type ProcessingType = 'remove-floor' | 'remove-wall'

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [processedImage, setProcessedImage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [processingType, setProcessingType] = useState<ProcessingType>('remove-floor')
  
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

      // Create preview URL
      const reader = new FileReader()
      reader.onload = (e) => {
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
      
      if (data.success) {
        setProcessedImage(data.result_base64)
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8">
      <div className="container mx-auto px-4 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            🏠 Room Segmentation Tool
          </h1>
          <p className="text-gray-600 text-lg">
            Upload a room image to automatically remove floors or walls
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
                  {isLoading ? 'Processing...' : (processingType === 'remove-floor' ? 'Remove Floor' : 'Remove Wall')}
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
                {processingType === 'remove-floor' ? 'Removing floor...' : 'Removing wall...'}
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
                  {processedImage ? (processingType === 'remove-floor' ? 'Room Without Floor' : 'Room Without Wall') : 'Result will appear here'}
                </h3>
                {processedImage && (
                  <button
                    onClick={() => downloadImage(
                      processedImage,
                      processingType === 'remove-floor' ? 'room-no-floor.png' : 'room-no-wall.png'
                    )}
                    className="bg-blue-500 hover:bg-blue-600 text-white text-sm px-3 py-1 rounded transition-colors duration-200"
                  >
                    Download
                  </button>
                )}
              </div>
              
              {processedImage ? (
                <div className="relative w-full h-80 rounded-lg overflow-hidden bg-gray-100">
                  <Image
                    src={processedImage}
                    alt={processingType === 'remove-floor' ? 'Room without floor' : 'Room without wall'}
                    fill
                    className="object-contain"
                  />
                </div>
              ) : (
                <div className="w-full h-80 rounded-lg bg-gray-50 border-2 border-dashed border-gray-300 flex items-center justify-center">
                  <p className="text-gray-500">
                    {processingType === 'remove-floor' ? 'Floor-free image will appear here' : 'Wall-free image will appear here'}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}