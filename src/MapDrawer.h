
#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include<pangolin/pangolin.h>
#include <mutex>
// for Sophus
#include <sophus/se3.h>
#include <sophus/so3.h>
// for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

using Eigen::Vector2d;
using Eigen::Vector3d;

using namespace std;

namespace soft {
    class MapDrawer {
    public:
        MapDrawer(float mCameraSize, float mCameraLineWidth, float mViewpointX,
                  float mViewpointY, float mViewpointZ, float mViewpointF, float fps);

        void SetCurrentCameraPose(const Sophus::SE3 &Twc);

        void Run();

        void addKeyframePos(Eigen::Vector3d);


    private:


        float mCameraSize;
        float mCameraLineWidth;
        Sophus::SE3 mCameraPose;

        double mT;

        float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
        std::mutex mMutexCamera;
        std::mutex mMutextwc;

        vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> twcs;

    private:
        void DrawCoordinateSystem();

        void DrawTrajectory();

        void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);

        void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

    };

}

#endif // MAPDRAWER_H
