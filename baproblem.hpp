#ifndef BAPROBLEM_HPP_
#define BAPROBLEM_HPP_

#include "parametersse3.hpp"

#include <iostream>

#define SQ(x) (((x)*(x)))

using namespace std; 

class Sample
{
public:
    static int uniform(int from, int to);
    static double uniform();
    static double gaussian(double sigma);
};

/// PoseBlockSize can only be
/// 7 (quaternion + translation vector) or
/// 6 (rotation vector + translation vector)
template <int PoseBlockSize>
class BAProblem
{
public:
    BAProblem(int pose_num_, int point_num_, double pix_noise_, bool useOrdering = false);

    void solve(ceres::Solver::Options &opt, ceres::Solver::Summary* sum);

    ceres::Problem problem;
    ceres::ParameterBlockOrdering* ordering = NULL;

    double rmse_pose(int pose_num_); 
    double rmse_point(int point_num_); 
    int g_point_num; 
    int g_pose_num;

protected:
    PosePointParametersBlock<PoseBlockSize> states;
    PosePointParametersBlock<PoseBlockSize> true_states;

};


template<int PoseBlockSize>
double BAProblem<PoseBlockSize>::rmse_point(int point_num_)
{
    double rmse = 0; 
    for(int i=0; i<point_num_; i++){
        Eigen::Map<Vector3d> true_point_i(true_states.point(i));
        Eigen::Map<Vector3d> noise_point_i(states.point(i));

        Vector3d diff_pt = true_point_i - noise_point_i; 
        rmse = rmse +  SQ(diff_pt.x()) + SQ(diff_pt.y()) + SQ(diff_pt.z()); 
    }

    return sqrt(rmse/point_num_); 
}

template<int PoseBlockSize>
double BAProblem<PoseBlockSize>::rmse_pose(int pose_num_)
{
    double rmse = 0; 
    for(int i=0; i<pose_num_; i++){

        SE3 true_pose_se3;
        SE3 est_pose_se3; 

        true_states.getPose(i, true_pose_se3.rotation(), true_pose_se3.translation());
        states.getPose(i, est_pose_se3.rotation(), est_pose_se3.translation()); 


        Vector3d diff_pt = est_pose_se3.translation() - true_pose_se3.translation(); 
        rmse = rmse +  SQ(diff_pt.x()) + SQ(diff_pt.y()) + SQ(diff_pt.z()); 
    }

    return sqrt(rmse/pose_num_); 
}


template<int PoseBlockSize>
BAProblem<PoseBlockSize>::BAProblem(int pose_num_, int point_num_, double pix_noise_, bool useOrdering)
{
    if(useOrdering)
        ordering = new ceres::ParameterBlockOrdering;

    int pose_num = pose_num_;
    int point_num = point_num_;
    double PIXEL_NOISE = pix_noise_;

    g_point_num = point_num; 
    g_pose_num = pose_num; 

    states.create(pose_num, point_num);
    true_states.create(pose_num, point_num);

    for (int i = 0; i < point_num; ++i)
    {
        Eigen::Map<Vector3d> true_pt(true_states.point(i));
        true_pt = Vector3d((Sample::uniform() - 0.5) * 3,
                           Sample::uniform() - 0.5,
                           Sample::uniform() + 3);
    }

    double focal_length = 1000.;
    double cx = 320.;
    double cy = 240.;
    CameraParameters cam(focal_length, cx, cy);

    for (int i = 0; i < pose_num; ++i)
    {
        Vector3d trans(i * 0.04 - 1., 0, 0);

        Eigen::Quaterniond q;
        q.setIdentity();
        true_states.setPose(i, q, trans);

        Vector3d noise_trans = trans +  Vector3d(Sample::gaussian(0.01),
                                                Sample::gaussian(0.01),
                                                Sample::gaussian(0.01));

        states.setPose(i, q, trans); //noise_trans trans

        problem.AddParameterBlock(states.pose(i), PoseBlockSize, new PoseSE3Parameterization<PoseBlockSize>());

        if(i < 2)
        {
            problem.SetParameterBlockConstant(states.pose(i));
        }
    }

    for (int i = 0; i < point_num; ++i)
    {
        Eigen::Map<Vector3d> true_point_i(true_states.point(i));
        Eigen::Map<Vector3d> noise_point_i(states.point(i));
        /*noise_point_i = true_point_i + Vector3d(Sample::gaussian(0.1),
                                                Sample::gaussian(0.1),
                                                Sample::gaussian(0.1));*/
        noise_point_i = true_point_i; 

        Vector2d z;
        SE3 true_pose_se3;

        int num_obs = 0;
        for (int j = 0; j < pose_num; ++j)
        {
            true_states.getPose(j, true_pose_se3.rotation(), true_pose_se3.translation());
            Vector3d point_cam = true_pose_se3.map(true_point_i);
            z = cam.cam_map(point_cam);
            if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
            {
                ++num_obs;
            }
        }
        if (num_obs >= 2)
        {
            problem.AddParameterBlock(states.point(i), 3);
            if(useOrdering)
                ordering->AddElementToGroup(states.point(i), 0);

            for (int j = 0; j < pose_num; ++j)
            {
                true_states.getPose(j, true_pose_se3.rotation(), true_pose_se3.translation());
                Vector3d point_cam = true_pose_se3.map(true_point_i);
                z = cam.cam_map(point_cam);

                if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
                {
                    z += Vector2d(Sample::gaussian(PIXEL_NOISE),
                                  Sample::gaussian(PIXEL_NOISE));
                    Eigen::Map<Vector3d> noise_point_i(states.point(i));
                    Vector3d point_cam = true_pose_se3.map(true_point_i);
                    double ux = (z.x() - cx)/focal_length; 
                    double vy = (z.y() - cy)/focal_length;
                    double d = point_cam.z(); 
                    noise_point_i = Vector3d(ux*d, vy*d, d); 

                    ceres::CostFunction* costFunc = new ReprojectionErrorSE3XYZ<PoseBlockSize>(focal_length, cx, cy, z[0], z[1]);
                    problem.AddResidualBlock(costFunc, NULL, states.pose(j), states.point(i));
                }
            }

           problem.SetParameterBlockConstant(states.point(i));

        }
    }

    if(useOrdering)
        for (int i = 0; i < pose_num; ++i)
        {
            ordering->AddElementToGroup(states.pose(i), 1);
        }

}


template<int PoseBlockSize>
void BAProblem<PoseBlockSize>::solve(ceres::Solver::Options& opt, ceres::Solver::Summary *sum)
{
    cout<<"before optimization rmse_point: "<<rmse_point(g_point_num)<<endl
        <<"rmse_pose: "<<rmse_pose(g_pose_num)<<endl; 

    if(ordering != NULL)
        opt.linear_solver_ordering.reset(ordering);
    ceres::Solve(opt, &problem, sum);
    cout<<"after optimization rmse_point: "<<rmse_point(g_point_num)<<endl
        <<"rmse_pose: "<<rmse_pose(g_pose_num)<<endl; 
}

#endif
