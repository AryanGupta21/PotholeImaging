#!/usr/bin/env python3
"""
Standalone Pothole 3D Analyzer
Processes pothole images using MiDaS and DPT models for depth estimation
Generates comprehensive analysis and visual outputs
"""

import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DPTImageProcessor, DPTForDepthEstimation
import json
import os
from datetime import datetime
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from scipy import ndimage
from skimage import feature, morphology, measure, segmentation
from skimage.segmentation import watershed
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist

class PotholeAnalyzer3D:
    def __init__(self):
        """Initialize both MiDaS and DPT models"""
        print("🚀 Initializing Pothole 3D Analyzer...")
        
        # Initialize DPT model
        print("📦 Loading DPT model...")
        self.dpt_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.dpt_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        
        # Initialize MiDaS model
        print("📦 Loading MiDaS model...")
        self.midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.midas_transform = self.midas_transforms.dpt_transform
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dpt_model.to(self.device)
        self.midas_model.to(self.device)
        
        # Model evaluation mode
        self.dpt_model.eval()
        self.midas_model.eval()
        
        print(f"✅ Models loaded successfully on {self.device}")
        
        # Constants for analysis - more realistic scaling
        self.PIXEL_TO_CM_RATIO = 0.05  # Default scale: 1 pixel = 0.5mm
        self.MAX_POTHOLE_DEPTH_CM = 15.0  # Realistic max depth for severe potholes
        self.MAX_POTHOLE_WIDTH_CM = 100.0  # Max realistic pothole width
        
    def estimate_depth_dpt(self, image_path):
        """Estimate depth using DPT model"""
        print("🔍 Running DPT depth estimation...")
        
        image = Image.open(image_path).convert('RGB')
        
        # Process with DPT
        inputs = self.dpt_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.dpt_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        depth_map = depth.squeeze().cpu().numpy()
        return self._normalize_depth(depth_map), image
    
    def estimate_depth_midas(self, image_path):
        """Estimate depth using MiDaS model"""
        print("🔍 Running MiDaS depth estimation...")
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transform for MiDaS
        input_tensor = self.midas_transform(img_rgb).to(self.device)
        
        # Prediction
        with torch.no_grad():
            prediction = self.midas_model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        return self._normalize_depth(depth_map), Image.fromarray(img_rgb)
    
    def _normalize_depth(self, depth_map):
        """Normalize depth map for consistent analysis"""
        # Remove extreme outliers
        depth_flat = depth_map.flatten()
        p5, p95 = np.percentile(depth_flat, [5, 95])
        depth_map = np.clip(depth_map, p5, p95)
        
        # Normalize to 0-1 range
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_map
    
    def detect_pothole_regions(self, depth_map, sensitivity=0.15):
        """Detect pothole regions in depth map with improved accuracy"""
        print(f"🎯 Detecting pothole regions (sensitivity: {sensitivity})...")
        
        height, width = depth_map.shape
        
        # Apply Gaussian filter to smooth the depth map
        smoothed_depth = ndimage.gaussian_filter(depth_map, sigma=2)
        
        # Find local minima (potential potholes are deeper/darker regions)
        # Invert depth map so potholes become local maxima
        inverted_depth = 1.0 - smoothed_depth
        
        # Use adaptive thresholding based on local statistics
        threshold_value = np.percentile(inverted_depth, 85 + (sensitivity * 50))
        
        # Create binary mask for potential potholes
        pothole_mask = inverted_depth > threshold_value
        
        # Clean up the mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        pothole_mask = cv2.morphologyEx(pothole_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        pothole_mask = cv2.morphologyEx(pothole_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(pothole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Advanced filtering of contours
        valid_contours = []
        min_area = (width * height) * 0.0005  # Minimum 0.05% of image
        max_area = (width * height) * 0.3     # Maximum 30% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size filter
            if area < min_area or area > max_area:
                continue
            
            # Shape filter - reject very elongated shapes (likely cracks, not potholes)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
            
            if aspect_ratio > 4:  # Reject very elongated shapes (reduced from 5)
                continue
            
            # Solidity filter - potholes should be reasonably solid shapes
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.4:  # Reject very non-convex shapes (increased from 0.3)
                continue
            
            # Position filter - avoid edges of image (often artifacts)
            center_x = x + w // 2
            center_y = y + h // 2
            
            margin = min(width, height) * 0.05  # Reduced margin to 5%
            if (center_x < margin or center_x > width - margin or 
                center_y < margin or center_y > height - margin):
                continue
            
            # Circularity filter - potholes are usually somewhat circular
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.3:  # Reject very non-circular shapes
                    continue
            
            # Depth verification - check if region is actually deeper
            mask = np.zeros(depth_map.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 1)
            region_depths = smoothed_depth[mask.astype(bool)]
            
            if len(region_depths) > 0:
                # Compare region depth with surrounding area
                # Create expanded mask for surrounding area
                kernel_expand = np.ones((15, 15), np.uint8)
                expanded_mask = cv2.dilate(mask, kernel_expand, iterations=1)
                surrounding_mask = expanded_mask - mask
                surrounding_depths = smoothed_depth[surrounding_mask.astype(bool)]
                
                if len(surrounding_depths) > 0:
                    region_mean_depth = np.mean(region_depths)
                    surrounding_mean_depth = np.mean(surrounding_depths)
                    
                    # Pothole should be deeper (higher normalized depth) than surroundings
                    depth_difference = region_mean_depth - surrounding_mean_depth
                    
                    if depth_difference < 0.03:  # Minimum depth difference (reduced threshold)
                        continue
            
            valid_contours.append(contour)
        
        # Create final mask from valid contours
        final_mask = np.zeros(depth_map.shape, dtype=np.uint8)
        if valid_contours:
            cv2.fillPoly(final_mask, valid_contours, 1)
        
        print(f"✅ Found {len(valid_contours)} valid potholes (filtered from {len(contours)} initial detections)")
        return final_mask.astype(bool), valid_contours
    
    def analyze_potholes(self, depth_map, contours, pixel_scale=None):
        """Analyze detected potholes and extract measurements"""
        print("📊 Analyzing pothole dimensions...")
        
        if not contours:
            return {
                "status": "no_potholes_detected",
                "message": "No significant depressions found"
            }
        
        if pixel_scale is None:
            pixel_scale = self._estimate_pixel_scale(depth_map.shape)
        
        potholes = []
        
        for i, contour in enumerate(contours):
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create mask for this pothole
            mask = np.zeros(depth_map.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 1)
            mask = mask.astype(bool)
            
            # Extract depth values
            pothole_depths = depth_map[mask]
            
            if len(pothole_depths) == 0:
                continue
            
            # Calculate measurements
            area_pixels = cv2.contourArea(contour)
            perimeter_pixels = cv2.arcLength(contour, True)
            
            # Physical dimensions with realistic bounds
            width_cm = min(w * pixel_scale, self.MAX_POTHOLE_WIDTH_CM)
            length_cm = min(h * pixel_scale, self.MAX_POTHOLE_WIDTH_CM)
            area_cm2 = min(area_pixels * (pixel_scale ** 2), self.MAX_POTHOLE_WIDTH_CM ** 2)
            perimeter_cm = perimeter_pixels * pixel_scale
            
            # Depth analysis with realistic scaling
            avg_depth_normalized = np.mean(pothole_depths)
            max_depth_normalized = np.max(pothole_depths)
            
            # Convert to physical depth with realistic bounds (0-15cm max)
            avg_depth_cm = min(avg_depth_normalized * self.MAX_POTHOLE_DEPTH_CM, self.MAX_POTHOLE_DEPTH_CM)
            max_depth_cm = min(max_depth_normalized * self.MAX_POTHOLE_DEPTH_CM, self.MAX_POTHOLE_DEPTH_CM)
            
            # Ensure minimum realistic values
            avg_depth_cm = max(avg_depth_cm, 0.5)  # Minimum 0.5cm depth
            max_depth_cm = max(max_depth_cm, avg_depth_cm)  # Max >= average
            
            # Volume calculation
            volume_cm3 = area_cm2 * avg_depth_cm
            
            # Shape analysis
            aspect_ratio = max(width_cm, length_cm) / min(width_cm, length_cm) if min(width_cm, length_cm) > 0 else 1
            circularity = (4 * np.pi * area_cm2) / (perimeter_cm ** 2) if perimeter_cm > 0 else 0
            
            # Severity classification
            severity = self._classify_severity(width_cm, length_cm, max_depth_cm, area_cm2)
            
            pothole_data = {
                "id": i + 1,
                "bounding_box": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },
                "dimensions": {
                    "width_cm": round(width_cm, 2),
                    "length_cm": round(length_cm, 2),
                    "area_cm2": round(area_cm2, 2),
                    "perimeter_cm": round(perimeter_cm, 2)
                },
                "depth_analysis": {
                    "avg_depth_cm": round(avg_depth_cm, 2),
                    "max_depth_cm": round(max_depth_cm, 2),
                    "volume_cm3": round(volume_cm3, 2)
                },
                "shape_analysis": {
                    "aspect_ratio": round(aspect_ratio, 2),
                    "circularity": round(circularity, 2),
                    "shape_type": "elongated" if aspect_ratio > 2 else "circular"
                },
                "severity": severity,
                "repair_priority": self._calculate_repair_priority(severity, area_cm2, max_depth_cm)
            }
            
            potholes.append(pothole_data)
        
        # Summary with realistic bounds
        total_area = sum(p["dimensions"]["area_cm2"] for p in potholes)
        total_volume = sum(p["depth_analysis"]["volume_cm3"] for p in potholes)
        
        # More realistic repair material calculation (asphalt density ~2.3 kg/L)
        repair_material_kg = (total_volume / 1000) * 2.3  # Convert cm³ to L, then to kg
        
        return {
            "status": "success",
            "pothole_count": len(potholes),
            "potholes": potholes,
            "summary": {
                "total_damaged_area_cm2": round(total_area, 2),
                "total_volume_cm3": round(total_volume, 2),
                "estimated_repair_material_kg": round(repair_material_kg, 2),
                "overall_severity": max([p["severity"]["level"] for p in potholes]) if potholes else 0
            },
            "pixel_scale_used": pixel_scale
        }
    
    def _estimate_pixel_scale(self, image_shape):
        """Estimate pixel to centimeter ratio with realistic bounds"""
        height, width = image_shape
        
        # More conservative estimation based on typical mobile phone camera
        # Assume image covers roughly 1-2 meters width of road when taken from ~1.5m height
        estimated_road_width_cm = 150  # 1.5 meters
        base_pixel_scale = estimated_road_width_cm / width
        
        # Clamp to reasonable bounds (0.02cm to 0.1cm per pixel)
        pixel_scale = np.clip(base_pixel_scale, 0.02, 0.1)
        
        return pixel_scale
    
    def _classify_severity(self, width_cm, length_cm, depth_cm, area_cm2):
        """Classify pothole severity with realistic thresholds"""
        severity_score = 0
        
        # Size factor (realistic thresholds)
        if area_cm2 > 400:  # > 20cm x 20cm (large)
            severity_score += 2
        elif area_cm2 > 100:  # > 10cm x 10cm (medium)
            severity_score += 1
        
        # Depth factor (most critical) - realistic depth ranges
        if depth_cm > 8:  # > 8cm is critical
            severity_score += 3
        elif depth_cm > 4:  # > 4cm is severe
            severity_score += 2
        elif depth_cm > 2:  # > 2cm is moderate
            severity_score += 1
        
        # Dimension factor - realistic size ranges
        max_dimension = max(width_cm, length_cm)
        if max_dimension > 50:  # > 50cm is large
            severity_score += 2
        elif max_dimension > 25:  # > 25cm is medium
            severity_score += 1
        
        # Classification with realistic descriptions
        if severity_score >= 6:
            return {"level": 4, "description": "Critical", "action": "Immediate repair required"}
        elif severity_score >= 4:
            return {"level": 3, "description": "Severe", "action": "Repair within 48 hours"}
        elif severity_score >= 2:
            return {"level": 2, "description": "Moderate", "action": "Repair within 1 week"}
        else:
            return {"level": 1, "description": "Minor", "action": "Monitor and repair within 1 month"}
    
    def _calculate_repair_priority(self, severity, area_cm2, depth_cm):
        """Calculate repair priority score"""
        base_score = severity["level"] * 20
        area_factor = min(area_cm2 / 100, 20)
        depth_factor = min(depth_cm * 4, 20)
        priority = min(base_score + area_factor + depth_factor, 100)
        return round(priority, 1)
    
    def create_visualizations(self, original_image, depth_map_dpt, depth_map_midas, 
                            contours, analysis_result, output_dir):
        """Create comprehensive visualizations"""
        print("🎨 Creating visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 0. Debug visualization (NEW - helps understand detection process)
        self.create_debug_visualization(original_image, depth_map_dpt, contours, output_dir)
        
        # 1. Depth comparison visualization
        self._create_depth_comparison(original_image, depth_map_dpt, depth_map_midas, output_dir)
        
        # 2. Pothole detection overlay
        self._create_detection_overlay(original_image, contours, analysis_result, output_dir)
        
        # 3. 3D surface plot
        self._create_3d_surface_plot(depth_map_dpt, output_dir)
        
        # 4. Analysis dashboard
        self._create_analysis_dashboard(analysis_result, output_dir)
        
        # 5. Interactive 3D plot with Plotly
        self._create_interactive_3d(depth_map_dpt, output_dir)
        
        print(f"✅ All visualizations saved to: {output_dir}")
    
    def _create_depth_comparison(self, original_image, depth_dpt, depth_midas, output_dir):
        """Create side-by-side depth comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # DPT depth
        im1 = axes[1].imshow(depth_dpt, cmap='plasma')
        axes[1].set_title('DPT Depth Estimation', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # MiDaS depth
        im2 = axes[2].imshow(depth_midas, cmap='plasma')
        axes[2].set_title('MiDaS Depth Estimation', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/depth_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_detection_overlay(self, original_image, contours, analysis_result, output_dir):
        """Create pothole detection overlay"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        ax.imshow(original_image)
        
        if analysis_result['status'] == 'success':
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
            
            for i, pothole in enumerate(analysis_result['potholes']):
                bbox = pothole['bounding_box']
                color = colors[i % len(colors)]
                
                # Draw bounding box
                rect = patches.Rectangle(
                    (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                    linewidth=3, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                severity = pothole['severity']['description']
                label = f"Pothole {pothole['id']}\n{severity}\n{pothole['dimensions']['width_cm']}×{pothole['dimensions']['length_cm']}cm"
                
                ax.text(bbox['x'], bbox['y'] - 10, label, 
                       fontsize=10, fontweight='bold', color=color,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_title('Pothole Detection Results', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/detection_overlay.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_3d_surface_plot(self, depth_map, output_dir):
        """Create 3D surface plot of the depth map"""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Subsample for performance
        step = max(1, depth_map.shape[0] // 100)
        y, x = np.mgrid[0:depth_map.shape[0]:step, 0:depth_map.shape[1]:step]
        z = depth_map[::step, ::step]
        
        # Create surface plot
        surf = ax.plot_surface(x, y, z, cmap='terrain', alpha=0.8)
        
        ax.set_title('3D Road Surface Reconstruction', fontsize=16, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Depth (normalized)')
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/3d_surface.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_analysis_dashboard(self, analysis_result, output_dir):
        """Create analysis dashboard with charts"""
        if analysis_result['status'] != 'success':
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        potholes = analysis_result['potholes']
        
        # 1. Severity distribution
        severity_counts = {}
        for pothole in potholes:
            severity = pothole['severity']['description']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if severity_counts:
            axes[0, 0].pie(severity_counts.values(), labels=severity_counts.keys(), 
                          autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Severity Distribution', fontweight='bold')
        
        # 2. Size vs Depth scatter
        if potholes:
            areas = [p['dimensions']['area_cm2'] for p in potholes]
            depths = [p['depth_analysis']['max_depth_cm'] for p in potholes]
            colors = [p['severity']['level'] for p in potholes]
            
            scatter = axes[0, 1].scatter(areas, depths, c=colors, cmap='Reds', s=100, alpha=0.7)
            axes[0, 1].set_xlabel('Area (cm²)')
            axes[0, 1].set_ylabel('Max Depth (cm)')
            axes[0, 1].set_title('Area vs Depth Analysis', fontweight='bold')
            plt.colorbar(scatter, ax=axes[0, 1], label='Severity Level')
        
        # 3. Repair priority bar chart
        if potholes:
            priorities = [p['repair_priority'] for p in potholes]
            pothole_ids = [f"Pothole {p['id']}" for p in potholes]
            
            bars = axes[1, 0].bar(pothole_ids, priorities, color='orange', alpha=0.7)
            axes[1, 0].set_ylabel('Repair Priority (0-100)')
            axes[1, 0].set_title('Repair Priority Ranking', fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, priority in zip(bars, priorities):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                               f'{priority}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Summary statistics
        summary = analysis_result['summary']
        stats_text = f"""
        Analysis Summary:
        
        Total Potholes: {analysis_result['pothole_count']}
        Total Damaged Area: {summary['total_damaged_area_cm2']} cm²
        Total Volume: {summary['total_volume_cm3']} cm³
        Repair Material Needed: {summary['estimated_repair_material_kg']} kg
        Overall Severity: {summary['overall_severity']}/4
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Statistics', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/analysis_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_debug_visualization(self, original_image, depth_map, contours, output_dir):
        """Create detailed debug visualization to understand detection process"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Raw depth map
        im1 = axes[0, 1].imshow(depth_map, cmap='plasma')
        axes[0, 1].set_title('Raw Depth Map', fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 3. Inverted depth (potholes as bright spots)
        inverted_depth = 1.0 - depth_map
        im2 = axes[0, 2].imshow(inverted_depth, cmap='hot')
        axes[0, 2].set_title('Inverted Depth (Potholes=Bright)', fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # 4. Thresholded mask
        threshold_value = np.percentile(inverted_depth, 85)
        binary_mask = inverted_depth > threshold_value
        axes[1, 0].imshow(binary_mask, cmap='gray')
        axes[1, 0].set_title(f'Binary Mask (threshold={threshold_value:.2f})', fontweight='bold')
        axes[1, 0].axis('off')
        
        # 5. Detected contours on original
        axes[1, 1].imshow(original_image)
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for i, contour in enumerate(contours):
            color = colors[i % len(colors)]
            x, y, w, h = cv2.boundingRect(contour)
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            axes[1, 1].add_patch(rect)
            
            # Add center point
            center_x, center_y = x + w//2, y + h//2
            axes[1, 1].plot(center_x, center_y, 'o', color=color, markersize=8)
            axes[1, 1].text(center_x, center_y-10, f'P{i+1}', color=color, fontweight='bold')
        
        axes[1, 1].set_title(f'Detected Potholes ({len(contours)} found)', fontweight='bold')
        axes[1, 1].axis('off')
        
        # 6. Depth profile analysis
        if contours:
            # Show depth profile of first detected pothole
            contour = contours[0]
            mask = np.zeros(depth_map.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 1)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract depth profile along the center line
            center_y_line = y + h // 2
            if center_y_line < depth_map.shape[0]:
                depth_profile = depth_map[center_y_line, x:x+w]
                axes[1, 2].plot(range(len(depth_profile)), depth_profile, 'b-', linewidth=2)
                axes[1, 2].set_title('Depth Profile (Center Line)', fontweight='bold')
                axes[1, 2].set_xlabel('X Position (pixels)')
                axes[1, 2].set_ylabel('Normalized Depth')
                axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No potholes detected', ha='center', va='center', 
                           transform=axes[1, 2].transAxes, fontsize=14)
            axes[1, 2].set_title('No Analysis Available', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/debug_detection_process.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🔍 Debug visualization saved: {output_dir}/debug_detection_process.png")
    
    def _create_interactive_3d(self, depth_map, output_dir):
        """Create interactive 3D visualization with Plotly"""
        try:
            # Subsample for performance
            step = max(1, depth_map.shape[0] // 50)
            y, x = np.mgrid[0:depth_map.shape[0]:step, 0:depth_map.shape[1]:step]
            z = depth_map[::step, ::step]
            
            fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
            
            fig.update_layout(
                title='Interactive 3D Road Surface',
                scene=dict(
                    xaxis_title='X (pixels)',
                    yaxis_title='Y (pixels)',
                    zaxis_title='Depth (normalized)'
                ),
                width=800,
                height=600
            )
            
            fig.write_html(f"{output_dir}/interactive_3d.html")
            print("📱 Interactive 3D visualization saved as HTML")
            
        except ImportError:
            print("⚠️  Plotly not available, skipping interactive 3D visualization")
        """Create interactive 3D visualization with Plotly"""
        try:
            # Subsample for performance
            step = max(1, depth_map.shape[0] // 50)
            y, x = np.mgrid[0:depth_map.shape[0]:step, 0:depth_map.shape[1]:step]
            z = depth_map[::step, ::step]
            
            fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
            
            fig.update_layout(
                title='Interactive 3D Road Surface',
                scene=dict(
                    xaxis_title='X (pixels)',
                    yaxis_title='Y (pixels)',
                    zaxis_title='Depth (normalized)'
                ),
                width=800,
                height=600
            )
            
            fig.write_html(f"{output_dir}/interactive_3d.html")
            print("📱 Interactive 3D visualization saved as HTML")
            
        except ImportError:
            print("⚠️  Plotly not available, skipping interactive 3D visualization")

def process_pothole_image(image_path, output_dir=None, sensitivity=0.15):
    """Main function to process a pothole image"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"pothole_analysis_{timestamp}"
    
    print(f"🔍 Processing: {image_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Initialize analyzer
    analyzer = PotholeAnalyzer3D()
    
    # Get depth estimates from both models
    depth_dpt, original_image = analyzer.estimate_depth_dpt(image_path)
    depth_midas, _ = analyzer.estimate_depth_midas(image_path)
    
    # Use DPT for main analysis (generally more accurate)
    pothole_mask, contours = analyzer.detect_pothole_regions(depth_dpt, sensitivity)
    analysis_result = analyzer.analyze_potholes(depth_dpt, contours)
    
    # Create visualizations
    analyzer.create_visualizations(
        original_image, depth_dpt, depth_midas, 
        contours, analysis_result, output_dir
    )
    
    # Save analysis results as JSON
    results_path = os.path.join(output_dir, "analysis_results.json")
    with open(results_path, 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("🔍 POTHOLE ANALYSIS COMPLETE")
    print("="*60)
    
    if analysis_result['status'] == 'success':
        print(f"✅ Potholes detected: {analysis_result['pothole_count']}")
        summary = analysis_result['summary']
        print(f"📊 Total damaged area: {summary['total_damaged_area_cm2']} cm²")
        print(f"🪣 Total volume: {summary['total_volume_cm3']} cm³")
        print(f"⚖️  Repair material needed: {summary['estimated_repair_material_kg']} kg")
        
        for pothole in analysis_result['potholes']:
            print(f"\n🕳️  Pothole #{pothole['id']}:")
            print(f"   📏 Size: {pothole['dimensions']['width_cm']}×{pothole['dimensions']['length_cm']} cm")
            print(f"   📊 Max depth: {pothole['depth_analysis']['max_depth_cm']} cm")
            print(f"   ⚠️  Severity: {pothole['severity']['description']}")
    else:
        print("✅ No potholes detected in the image")
    
    print(f"\n📁 All results saved to: {output_dir}")
    print("="*60)
    
    return analysis_result

if __name__ == "__main__":
    # Example usage
    image_path = "pothole_sample.jpg"  # Change this to your image path
    
    # Process the image
    result = process_pothole_image(image_path, sensitivity=0.15)
    
    # You can also process multiple images
    image_directory = "."  # Current directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    print("\n🔍 Looking for images in current directory...")
    for filename in os.listdir(image_directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            if filename != "pothole_sample.jpg":  # Skip if already processed
                print(f"\n📷 Found image: {filename}")
                try:
                    process_pothole_image(filename)
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")