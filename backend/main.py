from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor, Mask2FormerForUniversalSegmentation
import warnings
import io
import base64
from typing import Optional
import uvicorn
import os

warnings.filterwarnings("ignore")

app = FastAPI(title="Room Segmentation API", version="1.0.0")

# CORS middleware - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SegmentationModel:
    def __init__(self):
        self.device = self._get_device()
        self.processor = None
        self.model = None
        self.wall_processor = None
        self.wall_model = None
        self.wall_support = False
        self._load_models()
    
    def _get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    
    def _load_models(self):
        # First try to load the floor segmentation model (SegFormer)
        try:
            print("Loading SegFormer model for floor removal...")
            # Try offline first if cache exists
            try:
                self.processor = AutoImageProcessor.from_pretrained(
                    "nvidia/segformer-b2-finetuned-ade-512-512",
                    cache_dir="./model_cache",
                    local_files_only=True
                )
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b2-finetuned-ade-512-512",
                    cache_dir="./model_cache",
                    local_files_only=True
                ).to(self.device)
                print("SegFormer model loaded from cache!")
            except Exception as offline_error:
                print(f"Cache not found, trying online download: {offline_error}")
                # Try online if offline fails
                self.processor = AutoImageProcessor.from_pretrained(
                    "nvidia/segformer-b2-finetuned-ade-512-512",
                    cache_dir="./model_cache"
                )
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b2-finetuned-ade-512-512",
                    cache_dir="./model_cache"
                ).to(self.device)
                print("SegFormer model downloaded and loaded!")
            
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading SegFormer model: {e}")
            raise Exception(f"Failed to load floor segmentation model: {e}")
        
        # Now try to load the wall segmentation model (Mask2Former) - optional
        try:
            print("Loading Mask2Former model for wall removal...")
            # Try offline first if cache exists
            try:
                self.wall_processor = AutoImageProcessor.from_pretrained(
                    "facebook/mask2former-swin-large-ade-semantic",
                    cache_dir="./model_cache",
                    local_files_only=True
                )
                self.wall_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                    "facebook/mask2former-swin-large-ade-semantic",
                    cache_dir="./model_cache",
                    local_files_only=True
                ).to(self.device)
                print("Mask2Former model loaded from cache!")
                self.wall_support = True
            except Exception as offline_error:
                print(f"Wall model cache not found, trying online download: {offline_error}")
                # Try online if offline fails
                self.wall_processor = AutoImageProcessor.from_pretrained(
                    "facebook/mask2former-swin-large-ade-semantic",
                    cache_dir="./model_cache"
                )
                self.wall_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                    "facebook/mask2former-swin-large-ade-semantic",
                    cache_dir="./model_cache"
                ).to(self.device)
                print("Mask2Former model downloaded and loaded!")
                self.wall_support = True
            
            self.wall_model.eval()
            
        except Exception as e:
            print(f"Warning: Could not load wall segmentation model: {e}")
            print("Wall removal features will be disabled. Only floor removal will be available.")
            self.wall_support = False
    
    def get_room_without_wall(self, image: Image.Image) -> Image.Image:
        """
        Remove walls from room image and return image with transparent walls
        """
        if not self.wall_support:
            raise HTTPException(
                status_code=503, 
                detail="Wall removal feature is not available. The required model could not be loaded."
            )
        
        try:
            # Convert to RGB
            image_rgb = image.convert("RGB")
            
            # Validate image size
            if image_rgb.width < 100 or image_rgb.height < 100:
                raise ValueError("Image too small (minimum 100x100 pixels)")
            
            # Perform segmentation
            inputs = self.wall_processor(images=image_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.wall_model(**inputs)
            
            # Get segmentation map
            segmentation = self.wall_processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image_rgb.size[::-1]]
            )[0].cpu().numpy()
            
            # Find walls (class 0 in ADE20K)
            wall_mask = (segmentation == 0).astype(np.uint8)
            
            if np.sum(wall_mask) == 0:
                raise ValueError("No walls detected in the image")
            
            # Find largest connected wall component
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(wall_mask, connectivity=8)
            
            if num_labels <= 1:
                raise ValueError("No significant wall areas detected")
            
            # Get largest wall (excluding background)
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            largest_wall_mask = (labels == largest_idx).astype(np.uint8)
            
            # Refine mask with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            largest_wall_mask = cv2.morphologyEx(largest_wall_mask, cv2.MORPH_CLOSE, kernel)
            largest_wall_mask = cv2.morphologyEx(largest_wall_mask, cv2.MORPH_OPEN, kernel)
            
            # Smooth edges
            largest_wall_mask = cv2.GaussianBlur(largest_wall_mask.astype(np.float32), (5, 5), 0)
            largest_wall_mask = (largest_wall_mask > 0.5).astype(np.uint8)
            
            # Create RGBA image with transparent wall
            image_rgba = Image.new("RGBA", image_rgb.size)
            image_rgba.paste(image_rgb, (0, 0))
            
            # Apply transparency to wall area
            alpha = 255 * (1 - largest_wall_mask)
            alpha_image = Image.fromarray(alpha.astype(np.uint8), mode='L')
            image_rgba.putalpha(alpha_image)
            
            return image_rgba
            
        except Exception as e:
            print(f"Error processing image for wall removal: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    def get_room_without_floor(self, image: Image.Image) -> tuple[Image.Image, list]:
        """
        Get room image with floor removed (transparent) and return edge coordinates
        Returns: tuple (processed_image, edge_coordinates_list)
        """
        try:
            # Convert to RGB first for processing, then to RGBA for transparency
            image_rgb = image.convert("RGB")
            image_rgba = image.convert("RGBA")
            image_np = np.array(image_rgba)
            
            # Validate image size
            if image_np.shape[0] < 100 or image_np.shape[1] < 100:
                raise ValueError("Image too small (minimum 100x100 pixels)")
            
            # Run segmentation on RGB image
            inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            segmentation = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()
            segmentation_resized = cv2.resize(
                segmentation.astype(np.uint8), 
                (image_np.shape[1], image_np.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Try multiple floor class indices (ADE20K dataset)
            floor_classes = [3, 4, 5, 13, 52, 95]  # floor, earth, grass, road, rug, mat
            
            # Find the best floor class based on bottom half of image
            bottom_half = segmentation_resized[segmentation_resized.shape[0]//2:, :]
            bottom_unique, bottom_counts = np.unique(bottom_half, return_counts=True)
            
            # Select the floor class with most pixels in bottom half
            best_floor_class = 3  # Default fallback
            max_pixels = 0
            
            for floor_class in floor_classes:
                if floor_class in bottom_unique:
                    idx = np.where(bottom_unique == floor_class)[0]
                    if len(idx) > 0:
                        pixel_count = bottom_counts[idx[0]]
                        if pixel_count > max_pixels:
                            max_pixels = pixel_count
                            best_floor_class = floor_class
            
            # Create floor mask
            floor_mask = (segmentation_resized == best_floor_class).astype(np.uint8)
            
            # Clean up the mask if we found substantial floor area
            total_floor_pixels = np.sum(floor_mask)
            if total_floor_pixels > 100:
                kernel = np.ones((3,3), np.uint8)
                floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)
                floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)
            
            # Find edge coordinates where floor mask changes
            edge_coordinates = self._extract_floor_edges(floor_mask)
            
            # Apply mask to make floor transparent
            result_image = image_np.copy()
            result_image[:, :, 3] = (1 - floor_mask) * 255  # Set alpha channel
            
            # Convert back to PIL Image
            result_pil = Image.fromarray(result_image.astype(np.uint8), 'RGBA')
            
            return result_pil, edge_coordinates
            
        except Exception as e:
            print(f"Error processing image: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    def _extract_floor_edges(self, floor_mask: np.ndarray) -> list:
        """
        Extract edge coordinates where the floor mask changes from 0 to 1 or 1 to 0
        Returns a list of coordinate dictionaries with different edge types
        """
        try:
            # Method 1: Find contours (external boundaries)
            contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            edge_data = {
                "contours": [],
                "gradient_edges": [],
                "boundary_points": []
            }
            
            # Extract contour points
            for i, contour in enumerate(contours):
                # Skip very small contours
                if cv2.contourArea(contour) > 50:
                    # Simplify contour to reduce number of points
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    contour_points = []
                    for point in simplified_contour:
                        x, y = int(point[0][0]), int(point[0][1])
                        contour_points.append({"x": x, "y": y})
                    
                    edge_data["contours"].append({
                        "contour_id": i,
                        "points": contour_points,
                        "area": float(cv2.contourArea(contour))
                    })
            
            # Method 2: Gradient-based edge detection for more precise edges
            # Apply Gaussian blur to reduce noise
            blurred_mask = cv2.GaussianBlur(floor_mask.astype(np.float32), (5, 5), 1.0)
            
            # Calculate gradients
            grad_x = cv2.Sobel(blurred_mask, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred_mask, cv2.CV_32F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Threshold to find edge pixels
            edge_threshold = 0.1
            edge_pixels = gradient_magnitude > edge_threshold
            
            # Extract edge pixel coordinates
            edge_y, edge_x = np.where(edge_pixels)
            
            # Subsample edge points to reduce data size (take every nth point)
            step = max(1, len(edge_x) // 1000)  # Limit to ~1000 points
            sampled_indices = np.arange(0, len(edge_x), step)
            
            for idx in sampled_indices:
                x, y = int(edge_x[idx]), int(edge_y[idx])
                edge_data["gradient_edges"].append({
                    "x": x, 
                    "y": y, 
                    "magnitude": float(gradient_magnitude[y, x])
                })
            
            # Method 3: Boundary points using morphological operations
            # Create boundary by subtracting eroded mask from original
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(floor_mask, kernel, iterations=1)
            boundary_mask = floor_mask - eroded_mask
            
            # Extract boundary coordinates
            boundary_y, boundary_x = np.where(boundary_mask > 0)
            
            # Subsample boundary points
            step = max(1, len(boundary_x) // 500)  # Limit to ~500 points
            sampled_indices = np.arange(0, len(boundary_x), step)
            
            for idx in sampled_indices:
                x, y = int(boundary_x[idx]), int(boundary_y[idx])
                edge_data["boundary_points"].append({"x": x, "y": y})
            
            return edge_data
            
        except Exception as e:
            print(f"Error extracting edge coordinates: {e}")
            return {
                "contours": [],
                "gradient_edges": [],
                "boundary_points": [],
                "error": str(e)
            }

# Initialize model
segmentation_model = SegmentationModel()

@app.get("/")
async def root():
    return {
        "message": "Room Segmentation API is running",
        "floor_removal": True,
        "wall_removal": segmentation_model.wall_support
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "device": str(segmentation_model.device),
        "features": {
            "floor_removal": True,
            "wall_removal": segmentation_model.wall_support
        }
    }

@app.post("/remove-wall")
async def remove_wall_from_image(file: UploadFile = File(...)):
    """
    Endpoint to remove walls from room image and return image with transparent walls
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not segmentation_model.wall_support:
        raise HTTPException(
            status_code=503, 
            detail="Wall removal feature is not available. The required model could not be loaded."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get room image without walls
        result_image = segmentation_model.get_room_without_wall(image)
        
        # Convert result to bytes
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            io.BytesIO(img_byte_arr.read()),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=room_no_wall.png"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove-wall-base64")
async def remove_wall_base64(file: UploadFile = File(...)):
    """
    Endpoint to remove walls from room image and return base64 encoded result
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not segmentation_model.wall_support:
        raise HTTPException(
            status_code=503, 
            detail="Wall removal feature is not available. The required model could not be loaded."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get room image without walls
        result_image = segmentation_model.get_room_without_wall(image)
        
        # Convert result to base64
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
        
        return {
            "success": True,
            "result_base64": f"data:image/png;base64,{img_base64}",
            "message": "Wall removal completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove-floor")
async def remove_floor_from_image(file: UploadFile = File(...)):
    """
    Endpoint to remove floor from room image and return image with transparent floor
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get room image without floor
        result_image = segmentation_model.get_room_without_floor(image)
        
        # Convert result to bytes
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            io.BytesIO(img_byte_arr.read()),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=room_no_floor.png"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove-floor-base64")
async def remove_floor_base64(file: UploadFile = File(...)):
    """
    Endpoint to remove floor from room image and return base64 encoded result with edges
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get room image without floor and edge coordinates
        result_image, edges = segmentation_model.get_room_without_floor(image)
        
        # Convert result to base64
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
        
        return {
            "success": True,
            "result_base64": f"data:image/png;base64,{img_base64}",
            "message": "Floor removal completed successfully",
            "edges": edges  # Fixed: proper dictionary key-value syntax
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )