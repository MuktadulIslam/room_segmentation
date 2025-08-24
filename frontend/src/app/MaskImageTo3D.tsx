import React, { useState, useEffect } from 'react';
import { Canvas, } from '@react-three/fiber';
import * as THREE from 'three';

// TypeScript interfaces
interface MaskPoint {
    x: number;
    y: number;
    z: number;
}

interface MaskImageTo3DProps {
    maskImage?: string; // Base64 image data
    onPlaneGenerated?: (vertices: number) => void;
}

// Mask processing utilities
async function processMaskImage(imageUrl: string): Promise<{
    maskData: ImageData,
    points: MaskPoint[]
}> {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();

        if (!ctx) {
            reject(new Error('Could not get canvas context'));
            return;
        }

        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;

            // Draw image to canvas
            ctx.drawImage(img, 0, 0);

            // Get image data
            const maskData = ctx.getImageData(0, 0, canvas.width, canvas.height);

            // Create texture from the image
            const texture = new THREE.CanvasTexture(canvas);
            texture.wrapS = THREE.RepeatWrapping;
            texture.wrapT = THREE.RepeatWrapping;
            texture.flipY = true;  // Correct texture orientation

            // Extract white pixel positions
            const points: MaskPoint[] = [];
            const data = maskData.data;
            const threshold = 128;

            for (let y = 0; y < canvas.height; y++) {
                for (let x = 0; x < canvas.width; x++) {
                    const idx = (y * canvas.width + x) * 4;
                    const r = data[idx];

                    if (r > threshold) { // White pixel
                        points.push({
                            x: (x - canvas.width / 2) * 0.01,   // Scale and center
                            y: (y - canvas.height / 2) * 0.01,  // Correct Y coordinate (no flip)
                            z: 0
                        });
                    }
                }
            }

            resolve({
                maskData,
                points
            });
        };

        img.onerror = () => reject(new Error('Failed to load image'));
        img.src = imageUrl;
    });
}


function createTexturedPlane(width: number, height: number): THREE.PlaneGeometry {
    // Maintain original aspect ratio
    const aspectRatio = width / height;
    const baseSize = 5; // Base size for consistent scaling

    let planeWidth, planeHeight;
    if (aspectRatio > 1) {
        // Landscape
        planeWidth = baseSize;
        planeHeight = baseSize / aspectRatio;
    } else {
        // Portrait or square
        planeHeight = baseSize;
        planeWidth = baseSize * aspectRatio;
    }

    const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight, 1, 1);
    return geometry;
}


function DynamicPlane({ maskData }: { maskData: ImageData }) {
    const [processedData, setProcessedData] = useState<{
        geometry?: THREE.BufferGeometry | THREE.Group;
        material?: THREE.Material;
        texture?: THREE.Texture;
    }>({});

    // Process mask data into geometry
    useEffect(() => {
        const processGeometry = async () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            // Create temporary canvas from ImageData
            canvas.width = maskData.width;
            canvas.height = maskData.height;
            ctx.putImageData(maskData, 0, 0);

            const texture = new THREE.CanvasTexture(canvas);

            const geometry = createTexturedPlane(maskData.width, maskData.height);
            const material = new THREE.MeshPhongMaterial({
                map: texture,
                transparent: true,
            });


            setProcessedData({ geometry, material, texture });
        };

        processGeometry();
    }, [maskData]);

    if (!processedData.geometry || !processedData.material) {
        return null;
    }

    return (
        <group>
            {processedData.geometry instanceof THREE.Group ? (
                <primitive object={processedData.geometry} />
            ) : (
                <mesh
                    geometry={processedData.geometry}
                    material={processedData.material}
                    castShadow
                    receiveShadow
                />
            )}
        </group>
    );
};

// 3D Scene Component
function Scene3D({ maskData }: { maskData: ImageData | null }) {
    return (<>
        <ambientLight intensity={0.8} />
        <directionalLight
            position={[0, 0, 10]}
            intensity={0.4}
        />

        {maskData && (
            <DynamicPlane
                maskData={maskData}
            />
        )}
    </>);
};

// Main Component
export default function MaskImageTo3D({ maskImage, onPlaneGenerated }: MaskImageTo3DProps) {
    const [maskData, setMaskData] = useState<ImageData | null>(null);

    // Process mask image when it changes
    useEffect(() => {
        if (!maskImage) return;

        processMaskImage(maskImage)
            .then(({ maskData, points }) => {
                setMaskData(maskData);

                // Count white pixels
                let whitePixels = 0;
                const data = maskData.data;
                for (let i = 0; i < data.length; i += 4) {
                    if (data[i] > 128) whitePixels++;
                }

                if (onPlaneGenerated) {
                    onPlaneGenerated(points.length);
                }
            })
            .catch((error) => {
                console.error('Error processing mask image:', error);
            });

    }, [maskImage, onPlaneGenerated]);


    return (
        <div className="w-full max-w-7xl mx-auto p-6 bg-gray-100 min-h-screen">
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
                <div className="w-full h-[600px] border border-gray-300 rounded-lg overflow-hidden">
                    <Canvas
                        camera={{
                            position: [0, 0, 8],  // Simple front view
                            fov: 50,
                            up: [0, 1, 0]  // Ensure Y is up
                        }}
                        style={{ background: 'transparent' }}
                        gl={{ alpha: true, premultipliedAlpha: false }}
                    >
                        <Scene3D
                            maskData={maskData}
                        />
                    </Canvas>
                </div>
            </div>
        </div>
    );
};