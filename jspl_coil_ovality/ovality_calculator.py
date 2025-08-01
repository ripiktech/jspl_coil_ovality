"""
Ovality calculation module for steel coil analysis.
Provides methods to calculate ovality from segmentation masks.
"""

import cv2
import numpy as np
import traceback
from typing import Optional, Dict
from ripikutils.logsman import setup_logger


class OvalityCalculator:
    """Calculates ovality from a segmentation mask using multiple methods."""
    
    def __init__(self):
        self.logger = setup_logger(name=__name__)
    
    def calculate_multiple_methods(self, mask: np.ndarray) -> Dict[str, Optional[float]]:
        try:
            results = {}
            results['ellipse'] = self._calculate(mask)
            results['perimeter_area'] = self._calculate_perimeter_area_ovality(mask)
            results['bounding_rect'] = self._calculate_bounding_rect_ovality(mask)
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating multiple ovality methods: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'ellipse': None,
                'perimeter_area': None,
                'bounding_rect': None
            }
    
    def _calculate(self, mask: np.ndarray) -> Optional[float]:
        """Calculate ovality from a segmentation mask."""
        try:
            if mask is None or mask.size == 0:
                self.logger.warning("Empty or None mask provided")
                return None
            
            mask_uint8 = mask.astype(np.uint8)
            
            contours, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                self.logger.warning("No contours found in mask")
                return None

            contour = max(contours, key=cv2.contourArea)
            
            if len(contour) < 5:
                self.logger.warning("Contour has insufficient points for ellipse fitting")
                return None

            ellipse = cv2.fitEllipse(contour)
            (center_x, center_y), (minor_axis, major_axis), angle = ellipse
            
            if major_axis > 0:
                ovality = (major_axis - minor_axis) / major_axis
                self.logger.info(f"Ovality calculated: {ovality:.4f} (major: {major_axis:.1f}, minor: {minor_axis:.1f})")
                return ovality
            else:
                self.logger.warning("Invalid ellipse axes")
                return None
                
        except cv2.error as e:
            self.logger.error(f"Error fitting ellipse: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        except Exception as e:
            self.logger.error(f"Error calculating ovality: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _calculate_perimeter_area_ovality(self, mask: np.ndarray) -> Optional[float]:
        """Calculate ovality using perimeter to area ratio."""
        try:
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours: return None
                
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            if area > 0:
                # Circularity = 4 * pi * area / perimeter^2
                # Ovality is inverse of circularity
                circularity = 4 * np.pi * area / (perimeter ** 2)
                ovality = 1 - circularity
                return max(0, min(1, ovality))  # Clamp between 0 and 1
                
        except Exception as e:
            self.logger.error(f"Error in perimeter-area calculation: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _calculate_bounding_rect_ovality(self, mask: np.ndarray) -> Optional[float]:
        """Calculate ovality using bounding rectangle analysis."""
        try:
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None
                
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            
            if w > 0 and h > 0:
                # Ovality based on aspect ratio
                aspect_ratio = max(w, h) / min(w, h)
                ovality = (aspect_ratio - 1) / aspect_ratio
                return max(0, min(1, ovality))
                
        except Exception as e:
            self.logger.error(f"Error in bounding rectangle calculation: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None 