// components/3d/Floor3DControls.tsx
'use client'

interface Floor3DControlsProps {
  showWireframe: boolean
  setShowWireframe: (show: boolean) => void
  show3D: boolean
  setShow3D: (show: boolean) => void
}

export default function Floor3DControls({
  showWireframe,
  setShowWireframe,
  show3D,
  setShow3D
}: Floor3DControlsProps) {
  return (
    <div className="bg-white rounded-lg shadow-lg p-4 mb-4">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">3D Floor Plane Visualization</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Toggle 3D View */}
        <div className="flex items-center space-x-3">
          <label className="flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={show3D}
              onChange={(e) => setShow3D(e.target.checked)}
              className="sr-only"
            />
            <div className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
              show3D ? 'bg-blue-600' : 'bg-gray-300'
            }`}>
              <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                show3D ? 'translate-x-6' : 'translate-x-1'
              }`} />
            </div>
            <span className="ml-2 text-sm font-medium text-gray-700">
              Show 3D Floor Plane
            </span>
          </label>
        </div>

        {/* Wireframe Toggle */}
        <div className="flex items-center space-x-3">
          <label className="flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={showWireframe}
              onChange={(e) => setShowWireframe(e.target.checked)}
              className="sr-only"
              disabled={!show3D}
            />
            <div className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
              showWireframe && show3D ? 'bg-green-600' : 'bg-gray-300'
            } ${!show3D ? 'opacity-50 cursor-not-allowed' : ''}`}>
              <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                showWireframe && show3D ? 'translate-x-6' : 'translate-x-1'
              }`} />
            </div>
            <span className={`ml-2 text-sm font-medium ${show3D ? 'text-gray-700' : 'text-gray-400'}`}>
              Wireframe Mode
            </span>
          </label>
        </div>
      </div>

      {show3D && (
        <div className="mt-4 p-3 bg-blue-50 rounded-md">
          <p className="text-sm text-blue-700">
            <strong>View Controls:</strong> Click and drag to rotate • Scroll to zoom • 
            {showWireframe ? ' Wireframe shows the detected floor outline.' : ' Solid view shows a clean off-white floor plane.'}
          </p>
        </div>
      )}
    </div>
  )
}