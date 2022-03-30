// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_COLORED_TSDF_HPP
#define OPENCV_3D_COLORED_TSDF_HPP

#include "../precomp.hpp"
#include "tsdf_functions.hpp"

namespace cv
{

class ColoredTSDFVolume : public Volume
{
public:
    // dimension in voxels, size in meters
    ColoredTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist,
                      int _maxWeight, Point3i _resolution, bool zFirstMemOrder = true);
    virtual ~ColoredTSDFVolume() = default;

public:

    // Gets bounding box in volume coordinates with given precision:
    // VOLUME_UNIT - up to volume unit
    // VOXEL - up to voxel
    // returns (min_x, min_y, min_z, max_x, max_y, max_z) in volume coordinates
    virtual Vec6f getBoundingBox(int precision) const CV_OVERRIDE;

    // Enabels or disables new volume unit allocation during integration
    // Applicable for HashTSDF only
    virtual void setEnableGrowth(bool v) CV_OVERRIDE;
    // Returns if new volume units are allocated during integration or not
    // Applicable for HashTSDF only
    virtual bool getEnableGrowth() const CV_OVERRIDE;

    Point3i volResolution;
    WeightType maxWeight;

    Point3f volSize;
    float truncDist;
    Vec4i volDims;
    Vec8i neighbourCoords;
};

Ptr<ColoredTSDFVolume> makeColoredTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor,
                                             float _truncDist, int _maxWeight, Point3i _resolution);
Ptr<ColoredTSDFVolume> makeColoredTSDFVolume(const VolumeParams& _params);

}  // namespace cv

#endif // include guard
