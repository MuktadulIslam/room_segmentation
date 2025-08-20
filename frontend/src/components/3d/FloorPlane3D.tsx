// components/3d/FloorPlane3D.tsx
'use client'

import { Canvas } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

interface EdgeData {
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

interface FloorPlane3DProps {
  edges: EdgeData
  imageWidth: number
  imageHeight: number
  showWireframe?: boolean
}

function FloorMesh({ edges, imageWidth, imageHeight, showWireframe = false }: FloorPlane3DProps) {
  const meshRef = useRef<THREE.Mesh>(null)

  const { geometry, material } = useMemo(() => {
    // Create geometry from contours
    const shape = new THREE.Shape()
    
    if (edges.contours && edges.contours.length > 0) {
      // Use the largest contour (by area)
      const largestContour = edges.contours.reduce((prev, current) => 
        (prev.area > current.area) ? prev : current
      )
      
      if (largestContour.points.length > 2) {
        // Normalize coordinates to center around origin
        const normalizedPoints = largestContour.points.map(point => ({
          x: (point.x - imageWidth / 2) / 100, // Scale down
          y: -(point.y - imageHeight / 2) / 100 // Flip Y and scale down
        }))
        
        // Create shape from contour points
        shape.moveTo(normalizedPoints[0].x, normalizedPoints[0].y)
        for (let i = 1; i < normalizedPoints.length; i++) {
          shape.lineTo(normalizedPoints[i].x, normalizedPoints[i].y)
        }
        shape.closePath()
      }
    }
    
    // If no valid contours, create a fallback rectangle
    if (edges.contours.length === 0 || edges.contours[0].points.length < 3) {
      const width = imageWidth / 200
      const height = imageHeight / 200
      shape.moveTo(-width/2, -height/2)
      shape.lineTo(width/2, -height/2)
      shape.lineTo(width/2, height/2)
      shape.lineTo(-width/2, height/2)
      shape.closePath()
    }

    // Create a flat plane geometry from the shape
    const shapeGeometry = new THREE.ShapeGeometry(shape)
    
    // Create off-white material
    const planeMaterial = new THREE.MeshStandardMaterial({
      color: showWireframe ? 0x666666 : 0xf8f8f0, // Off-white color
      wireframe: showWireframe,
      side: THREE.DoubleSide,
      transparent: false,
      opacity: 1.0
    })

    return { geometry: shapeGeometry, material: planeMaterial }
  }, [edges, imageWidth, imageHeight, showWireframe])

  return (
    <mesh 
      ref={meshRef} 
      geometry={geometry} 
      material={material} 
      position={[0, 0, 0]}
      rotation={[-Math.PI / 2, 0, 0]} // Rotate to lay flat
    />
  )
}

export default function FloorPlane3D({ 
  edges, 
  imageWidth, 
  imageHeight, 
  showWireframe = false
}: FloorPlane3DProps) {
  return (
    <div className="w-full h-96 bg-gray-100 rounded-lg overflow-hidden">
      <Canvas>
        <PerspectiveCamera makeDefault position={[0, 8, 8]} />
        <OrbitControls 
          enableDamping 
          dampingFactor={0.05}
          minPolarAngle={0}
          maxPolarAngle={Math.PI / 2}
          enableZoom={true}
          enablePan={true}
        />
        
        {/* Lighting setup for clean appearance */}
        <ambientLight intensity={0.8} />
        <directionalLight 
          position={[10, 10, 5]} 
          intensity={0.5}
          castShadow
        />
        
        {/* Simple grid helper for reference */}
        <gridHelper 
          args={[20, 20, 0xcccccc, 0xcccccc]} 
          position={[0, -0.1, 0]} 
        />
        
        {/* Floor plane */}
        <FloorMesh 
          edges={edges}
          imageWidth={imageWidth}
          imageHeight={imageHeight}
          showWireframe={showWireframe}
        />
      </Canvas>
    </div>
  )
}