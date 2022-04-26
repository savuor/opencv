// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include "sparse_block_matrix.hpp"
#include "opencv2/3d/detail/optimizer.hpp"

namespace cv
{
namespace detail
{

#if defined(HAVE_EIGEN)

// matrix form of conjugation
static const cv::Matx44d M_Conj{ 1,  0,  0,  0,
                                 0, -1,  0,  0,
                                 0,  0, -1,  0,
                                 0,  0,  0, -1 };

// matrix form of quaternion multiplication from left side
static inline cv::Matx44d m_left(cv::Quatd q)
{
    // M_left(a)* V(b) =
    //    = (I_4 * a0 + [ 0 | -av    [    0 | 0_1x3
    //                   av | 0_3] +  0_3x1 | skew(av)]) * V(b)

    double w = q.w, x = q.x, y = q.y, z = q.z;
    return { w, -x, -y, -z,
             x,  w, -z,  y,
             y,  z,  w, -x,
             z, -y,  x,  w };
}

// matrix form of quaternion multiplication from right side
static inline cv::Matx44d m_right(cv::Quatd q)
{
    // M_right(b)* V(a) =
    //    = (I_4 * b0 + [ 0 | -bv    [    0 | 0_1x3
    //                   bv | 0_3] +  0_3x1 | skew(-bv)]) * V(a)

    double w = q.w, x = q.x, y = q.y, z = q.z;
    return { w, -x, -y, -z,
             x,  w,  z, -y,
             y, -z,  w,  x,
             z,  y, -x,  w };
}

// jacobian of quaternionic (exp(x)*q) : R_3 -> H near x == 0
static inline cv::Matx43d expQuatJacobian(cv::Quatd q)
{
    double w = q.w, x = q.x, y = q.y, z = q.z;
    return cv::Matx43d(-x, -y, -z,
                        w,  z, -y,
                       -z,  w,  x,
                        y, -x,  w);
}

// concatenate matrices vertically
template<typename _Tp, int m, int n, int k> static inline
cv::Matx<_Tp, m + k, n> concatVert(const cv::Matx<_Tp, m, n>& a, const cv::Matx<_Tp, k, n>& b)
{
    cv::Matx<_Tp, m + k, n> res;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(i, j) = a(i, j);
        }
    }
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(m + i, j) = b(i, j);
        }
    }
    return res;
}

// concatenate matrices horizontally
template<typename _Tp, int m, int n, int k> static inline
cv::Matx<_Tp, m, n + k> concatHor(const cv::Matx<_Tp, m, n>& a, const cv::Matx<_Tp, m, k>& b)
{
    cv::Matx<_Tp, m, n + k> res;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(i, j) = a(i, j);
        }
    }
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            res(i, n + j) = b(i, j);
        }
    }
    return res;
}


static double median(std::vector<double>& v)
{
    size_t n = v.size() / 2;
    if (n == 0) return 0;

    std::nth_element(v.begin(), v.begin() + n, v.end());
    double vn = v[n];

    if (n % 2 == 0)
    {
        std::nth_element(v.begin(), v.begin() + n - 1, v.end());
        return (vn + v[n - 1]) / 2.0;
    }
    else
    {
        return vn;
    }
}


// median absolute deviation
static double madEstimate(const std::vector<double>& arr, double& med)
{
    // STD to MAD scale
    const double MAD_SCALE = 1.4826;
    std::vector<double> v = arr;
    med = median(v);
    std::for_each(v.begin(), v.end(), [med](double& x) {x = std::abs(x - med); });
    return MAD_SCALE * median(v);
}


// average standard deviation
// faster than MAD but not so robust
static double stdEstimate(const std::vector<double>& arr, double avg)
{
    Scalar mean, stddev;
    meanStdDev(arr, mean, stddev);
    avg = mean[0];
    return stddev[0];
}


/*
static double tukeyWeight(double v, double sigma = 1.0)
{
    v /= sigma;
    const double b2 = TUKEY_B * TUKEY_B;
    if (std::abs(v) <= TUKEY_B)
    {
        double y = 1.0 - (v * v) / b2;
        return y * y;
    }
    else return 0;
}
*/


// Works the same as above but takes squared norm
// Can be optimized in the future to interpolated LUTs
static double tukeyWeightSq(double vv, double sigma = 1.0)
{
    const double TUKEY_B = 4.6851;
    const double b2 = TUKEY_B * TUKEY_B;
    const double b2inv = 1.0 / b2;
    double vn = vv / (sigma * sigma);
    if (vn <= b2)
    {
        double y = 1.0 - vn * b2inv;
        return y * y;
    }
    else return 0;
}

/*
static double huberWeight(double vnorm, double sigma = 1.0)
{
    const double HUBER_K = 1.345;
    if (std::abs(sigma) < 0.001) return 0;
    double x = (double)std::abs(vnorm / sigma);
    return (x > HUBER_K) ? HUBER_K / x : 1.0;
}
*/


// Works the same as above but takes squared norm
// Can be optimized in the future to interpolated LUTs
static double huberWeightSq(double vnorm2, double sigma = 1.0)
{
    const double HUBER_K = 1.345;
    const double h2 = HUBER_K * HUBER_K;
    double vn = vnorm2 / (sigma * sigma);
    return (vn > h2) ? std::sqrt( h2 / vn ) : 1.0;
}


class PoseGraphImpl;
class PoseGraphLevMarqBackend;

class PoseGraphLevMarq : public LevMarqBase
{
public:
    PoseGraphLevMarq(PoseGraphImpl* pg, const LevMarq::Settings& settings_ = LevMarq::Settings()) :
        LevMarqBase(makePtr<PoseGraphLevMarqBackend>(pg), settings_)
    { }
};


class PoseGraphImpl : public PoseGraph
{
public:
    struct Pose3d
    {
        Vec3d t;
        Quatd q;

        Pose3d() : t(), q(1, 0, 0, 0) { }

        Pose3d(const Matx33d& rotation, const Vec3d& translation)
            : t(translation), q(Quatd::createFromRotMat(rotation).normalize())
        { }

        explicit Pose3d(const Matx44d& pose) :
            Pose3d(pose.get_minor<3, 3>(0, 0), Vec3d(pose(0, 3), pose(1, 3), pose(2, 3)))
        { }

        inline Pose3d operator*(const Pose3d& otherPose) const
        {
            Pose3d out(*this);
            out.t += q.toRotMat3x3(QUAT_ASSUME_UNIT) * otherPose.t;
            out.q = out.q * otherPose.q;
            return out;
        }

        Affine3d getAffine() const
        {
            return Affine3d(q.toRotMat3x3(QUAT_ASSUME_UNIT), t);
        }

        inline Pose3d inverse() const
        {
            Pose3d out;
            out.q = q.conjugate();
            out.t = -(out.q.toRotMat3x3(QUAT_ASSUME_UNIT) * t);
            return out;
        }

        inline void normalizeRotation()
        {
            q = q.normalize();
        }

        // jacobian of exponential (exp(x)* q) : R_6->SE(3) near x == 0
        inline Matx<double, 7, 6> expJacobian()
        {
            Matx43d qj = expQuatJacobian(q);
            // x node layout is (rot_x, rot_y, rot_z, trans_x, trans_y, trans_z)
            // pose layout is (q_w, q_x, q_y, q_z, trans_x, trans_y, trans_z)
            return concatVert(concatHor(qj, Matx43d()),
                              concatHor(Matx33d(), Matx33d::eye()));
        }

        inline Pose3d oplus(const Vec6d dx)
        {
            Vec3d deltaRot(dx[0], dx[1], dx[2]), deltaTrans(dx[3], dx[4], dx[5]);
            Pose3d p;
            p.q = Quatd(0, deltaRot[0], deltaRot[1], deltaRot[2]).exp() * this->q;
            p.t = this->t + deltaTrans;
            return p;
        }
    };

    /*! \class GraphNode
     *  \brief Defines a node/variable that is optimizable in a posegraph
     *
     *  Detailed description
     */
    struct Node
    {
    public:
        explicit Node(size_t _nodeId = -1, const Affine3d& _pose = Affine3d())
            : id(_nodeId), isFixed(false), pose(_pose.rotation(), _pose.translation()),
            inNodes(), outNodes(), inEdges(), outEdges()
        { }

        Affine3d getPose() const
        {
            return pose.getAffine();
        }
        void setPose(const Affine3d& _pose)
        {
            pose = Pose3d(_pose.rotation(), _pose.translation());
        }

    public:
        size_t id;
        bool isFixed;
        Pose3d pose;

        // nodes that start edges which terminate in the node
        std::unordered_set<size_t> inNodes;
        // nodes that terminate edges which start from the node
        std::unordered_set<size_t> outNodes;
        // edges that go to the node
        std::unordered_set<size_t> inEdges;
        // edges that go from the node
        std::unordered_set<size_t> outEdges;
    };

    /*! \class PoseGraphEdge
     *  \brief Defines the constraints between two PoseGraphNodes
     *
     *  Detailed description
     */
    struct Edge
    {
    public:
        explicit Edge(size_t _sourceNodeId, size_t _targetNodeId, const Affine3f& _transformation,
                      const Matx66f& _information = Matx66f::eye());

        bool operator==(const Edge& edge)
        {
            if ((edge.sourceNodeId == sourceNodeId && edge.targetNodeId == targetNodeId) ||
                (edge.sourceNodeId == targetNodeId && edge.targetNodeId == sourceNodeId))
                return true;
            return false;
        }

    public:
        size_t sourceNodeId;
        size_t targetNodeId;
        Pose3d pose;
        Matx66f sqrtInfo;
    };

    PoseGraphImpl(int rf = ROBUST_DISABLED, int ef = ERROR_RIGHT) :
        robustFlags(rf), errorApplyFlags(ef), nodes(), edges(), lm()
    {
        CV_Assert(rf == ROBUST_DISABLED || (((rf | ROBUST_TUKEY) ^ (rf | ROBUST_HUBER)) && ((rf | ROBUST_STD) ^ (rf | ROBUST_MAD))));
    }

    virtual ~PoseGraphImpl() CV_OVERRIDE
    { }

    // Node may have any id >= 0
    virtual void addNode(size_t _nodeId, const Affine3d& _pose, int flags) CV_OVERRIDE;
    virtual bool isNodeExist(size_t nodeId) const CV_OVERRIDE
    {
        return (nodes.find(nodeId) != nodes.end());
    }

    virtual bool setNodeFixed(size_t nodeId, bool fixed) CV_OVERRIDE
    {
        // Discard optimizer since it keeps node-to-variable correspondence
        // It'll be created again at optimize() call
        lm.reset();

        auto it = nodes.find(nodeId);
        if (it != nodes.end())
        {
            it->second.isFixed = fixed;
            return true;
        }
        else
            return false;
    }

    virtual bool isNodeFixed(size_t nodeId) const CV_OVERRIDE
    {
        auto it = nodes.find(nodeId);
        if (it != nodes.end())
            return it->second.isFixed;
        else
            return false;
    }

    virtual Affine3d getNodePose(size_t nodeId) const CV_OVERRIDE
    {
        auto it = nodes.find(nodeId);
        if (it != nodes.end())
            return it->second.getPose();
        else
            return Affine3d();
    }

    virtual std::unordered_set<size_t> getInNodes(size_t nodeId) const CV_OVERRIDE
    {
        std::unordered_set<size_t> res;
        auto it = nodes.find(nodeId);
        if (it != nodes.end())
        {
            for (auto ii : it->second.inNodes)
            {
                res.insert(ii);
            }
        }
        return res;
    }


    virtual std::unordered_set<size_t> getOutNodes(size_t nodeId) const CV_OVERRIDE
    {
        std::unordered_set<size_t> res;
        auto it = nodes.find(nodeId);
        if (it != nodes.end())
        {
            for (auto ii : it->second.outNodes)
            {
                res.insert(ii);
            }
        }
        return res;
    }


    virtual std::unordered_set<size_t> getInEdges(size_t nodeId) const CV_OVERRIDE
    {
        std::unordered_set<size_t> res;
        auto it = nodes.find(nodeId);
        if (it != nodes.end())
        {
            for (auto ii : it->second.inEdges)
            {
                res.insert(ii);
            }
        }
        return res;
    }


    virtual std::unordered_set<size_t> getOutEdges(size_t nodeId) const CV_OVERRIDE
    {
        std::unordered_set<size_t> res;
        auto it = nodes.find(nodeId);
        if (it != nodes.end())
        {
            for (auto ii : it->second.outEdges)
            {
                res.insert(ii);
            }
        }
        return res;
    }

    virtual std::vector<size_t> getNodesIds() const CV_OVERRIDE
    {
        std::vector<size_t> ids;
        for (const auto& it : nodes)
        {
            ids.push_back(it.first);
        }
        return ids;
    }

    virtual size_t getNumNodes() const CV_OVERRIDE
    {
        return nodes.size();
    }

    // Edges have consequent indices starting from 0
    virtual void addEdge(size_t _sourceNodeId, size_t _targetNodeId, const Affine3f& _transformation,
                         const Matx66f& _information = Matx66f::eye()) CV_OVERRIDE;

    virtual size_t getEdgeStart(size_t i) const CV_OVERRIDE
    {
        return edges[i].sourceNodeId;
    }

    virtual size_t getEdgeEnd(size_t i) const CV_OVERRIDE
    {
        return edges[i].targetNodeId;
    }

    virtual Affine3d getEdgePose(size_t i) const CV_OVERRIDE
    {
        return edges[i].pose.getAffine();
    }

    virtual void setEdgePose(size_t i, Affine3d pose) CV_OVERRIDE
    {
        edges[i].pose = Pose3d(pose.matrix);
    }

    virtual Matx66f getEdgeInfo(size_t i) const CV_OVERRIDE
    {
        Matx66f s = edges[i].sqrtInfo;
        return s * s;
    }

    virtual size_t getNumEdges() const CV_OVERRIDE
    {
        return edges.size();
    }

    // checks if graph is connected and each edge connects exactly 2 nodes
    virtual bool isValid() const CV_OVERRIDE;

    // calculate cost function based on current nodes parameters
    virtual double calcEnergy() const CV_OVERRIDE;

    // calculate cost function based on provided nodes parameters
    double calcEnergyNodes(const std::map<size_t, Node>& newNodes) const;

    // creates an optimizer
    virtual Ptr<LevMarqBase> createOptimizer(const LevMarq::Settings& settings) CV_OVERRIDE
    {
        lm = makePtr<PoseGraphLevMarq>(this, settings);

        return lm;
    }

    // creates an optimizer
    virtual Ptr<LevMarqBase> createOptimizer() CV_OVERRIDE
    {
        lm = makePtr<PoseGraphLevMarq>(this, LevMarq::Settings()
                                             .setMaxIterations(100)
                                             .setCheckRelEnergyChange(true)
                                             .setRelEnergyDeltaTolerance(1e-6)
                                             .setGeodesic(true));

        return lm;
    }

    // Returns number of iterations elapsed or -1 if max number of iterations was reached or failed to optimize
    virtual LevMarq::Report optimize() CV_OVERRIDE;

    int robustFlags, errorApplyFlags;

    std::map<size_t, Node> nodes;
    std::vector<Edge> edges;

    Ptr<PoseGraphLevMarq> lm;
};


void PoseGraphImpl::addNode(size_t _nodeId, const Affine3d& _pose, int flags)
{
    // Discard optimizer since it keeps node-to-variable correspondence
    // It'll be created again at optimize() call
    lm.reset();

    Node node(_nodeId, _pose);
    node.isFixed = bool(flags & NODE_FIXED);

    size_t id = node.id;
    const auto& it = nodes.find(id);
    if (it != nodes.end())
    {
        // Duplicated node is replaced
        std::cout << "duplicated node, id=" << id << std::endl;
        nodes.insert(it, { id, node });
    }
    else
    {
        nodes.insert({ id, node });
    }
}

// Edges have consequent indices starting from 0
void PoseGraphImpl::addEdge(size_t _sourceNodeId, size_t _targetNodeId, const Affine3f& _transformation,
                            const Matx66f& _information)
{
    // Discard optimizer since it keeps node-to-variable correspondence and other graph data
    // It'll be created again at optimize() call
    lm.reset();

    Edge e(_sourceNodeId, _targetNodeId, _transformation, _information);

    bool srcFound = nodes.find(e.sourceNodeId) != nodes.end();
    bool dstFound = nodes.find(e.targetNodeId) != nodes.end();
    if (srcFound && dstFound)
    {
        // Multiedges are allowed
        Node& dst = nodes[e.targetNodeId];
        dst.inNodes.insert(e.sourceNodeId);
        dst.inEdges.insert(edges.size());
        Node& src = nodes[e.sourceNodeId];
        src.outNodes.insert(e.targetNodeId);
        src.outEdges.insert(edges.size());
        edges.push_back(e);
    }
    else if (!srcFound)
    {
        CV_Error(cv::Error::Code::StsBadArg, "Source node not found");
    }
    else if (!dstFound)
    {
        CV_Error(cv::Error::Code::StsBadArg, "Target node not found");
    }
}


// Cholesky decomposition of symmetrical 6x6 matrix
static inline cv::Matx66d llt6(Matx66d m)
{
    Matx66d L;
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < (i + 1); j++)
        {
            double sum = 0;
            for (int k = 0; k < j; k++)
                sum += L(i, k) * L(j, k);

            if (i == j)
                L(i, i) = sqrt(m(i, i) - sum);
            else
                L(i, j) = (1.0 / L(j, j) * (m(i, j) - sum));
        }
    }
    return L;
}

PoseGraphImpl::Edge::Edge(size_t _sourceNodeId, size_t _targetNodeId, const Affine3f& _transformation,
                          const Matx66f& _information) :
                          sourceNodeId(_sourceNodeId),
                          targetNodeId(_targetNodeId),
                          pose(_transformation.rotation(), _transformation.translation()),
                          sqrtInfo(llt6(_information))
{ }


bool PoseGraphImpl::isValid() const
{
    // 1. All non-fixed nodes should be connected (in any direction)
    // 2. There should be at least one fixed node in connected part

    size_t numNodes = getNumNodes();
    size_t numEdges = getNumEdges();

    if (!numNodes)
    {
        CV_LOG_INFO(NULL, "PoseGraph contains no nodes, skipping optimization");
        return false;
    }

    if (!numNodes || !numEdges)
    {
        CV_LOG_INFO(NULL, "PoseGraph contains no edges, skipping optimization");
        return false;
    }

    std::unordered_set<size_t> notFixedNodes;
    for (const auto& n : nodes)
    {
        if (!n.second.isFixed)
            notFixedNodes.insert(n.first);
    }

    if (notFixedNodes.empty())
    {
        CV_LOG_INFO(NULL, "PoseGraph contains no non-constant nodes, skipping optimization");
        return false;
    }

    std::unordered_set<size_t> nodesVisited;
    std::vector<size_t> nodesToVisit;

    // Take first non-constant node
    nodesToVisit.push_back(*notFixedNodes.begin());

    int nFixed = 0;
    while (!nodesToVisit.empty())
    {
        size_t currNodeId = nodesToVisit.back();
        nodesToVisit.pop_back();
        nodesVisited.insert(currNodeId);

        const Node& node = nodes.at(currNodeId);
        if (node.isFixed)
            nFixed++;

        std::vector<size_t> togo;
        for (auto s : node.inNodes)
        {
            togo.push_back(s);
        }
        for (auto s : node.outNodes)
        {
            togo.push_back(s);
        }

        for (auto s : togo)
        {
            if (nodesVisited.find(s) == nodesVisited.end())
            {
                nodesToVisit.push_back(s);
            }
        }
    }

    bool allNonFixedNodesVisited = true;
    for (auto nf : notFixedNodes)
    {
        if (nodesVisited.find(nf) == nodesVisited.end())
        {
            allNonFixedNodesVisited = false; break;
        }
    }

    if (!allNonFixedNodesVisited)
    {
        CV_LOG_INFO(NULL, "Not all non-fixed nodes are connected");
        return false;
    }

    if (!nFixed)
    {
        CV_LOG_INFO(NULL, "There should be at least one fixed node in connected part");
        return false;
    }

    return true;
}


//////////////////////////
// Optimization itself //
////////////////////////

static inline double poseError(Quatd sourceQuat, Vec3d sourceTrans, Quatd targetQuat, Vec3d targetTrans,
                               Quatd rotMeasured, Vec3d transMeasured, Matx66d sqrtInfoMatrix,
                               bool applyFromLeft, bool needJacobians,
                               Matx<double, 6, 4>& sqj, Matx<double, 6, 3>& stj,
                               Matx<double, 6, 4>& tqj, Matx<double, 6, 3>& ttj,
                               Vec6d& res)
{
    if (applyFromLeft)
    {
        // err_r = 2*Im(measure_r * source_r * conj(target_r))
        // err_t = target_t - measure_r * source_t * conj(measure_r) - measure_t
        // DISCARDED:
        // err_t = conj(measure_r) * (target_t - measure_t) * measure_r - source_t

        Quatd relativeQuat = targetQuat * sourceQuat.conjugate();

        //Quatd deltaRot = rotMeasured * sourceQuat * targetQuat.conjugate();
        Quatd deltaRot = rotMeasured * relativeQuat.conjugate();
        Vec3d relativeTrans = targetTrans - rotMeasured.toRotMat3x3(QUAT_ASSUME_UNIT) * sourceTrans;

        //Vec3d terr = rotMeasured.toRotMat3x3(QUAT_ASSUME_UNIT) * (targetTrans - transMeasured) - sourceTrans;
        Vec3d terr = relativeTrans - transMeasured;
        Vec3d rerr = 2.0 * Vec3d(deltaRot.x, deltaRot.y, deltaRot.z);
        Vec6d rterr(terr[0], terr[1], terr[2], rerr[0], rerr[1], rerr[2]);

        res = sqrtInfoMatrix * rterr;

        if (needJacobians)
        {
            // d(err_r) = d(2*Im(measure_r * source_r * conj(target_r))) = <measure_r is constant> =
            // 2*Im( measure_r * (d(source_r) * conj(target_r) + source_r * conj(d(target_r))) )
            // d(target_r) == 0:
            // # d(err_r) = 2*Im( measure_r * d(source_r) * conj(target_r) )
            // # V(d(err_r)) = 2 * M_Im * M_right(conj(target_r)) * M_left(measure_r) * V(d(source_r))
            // # d(err_r) / d(source_r) = 2 * M_Im * M_right(conj(target_r)) * M_left(measure_r)
            Matx34d drdsq = 2.0 * (m_right(targetQuat.conjugate()) * m_left(rotMeasured)).get_minor<3, 4>(1, 0);

            // d(source_r) == 0:
            // # d(err_r) = 2*Im( measure_r * source_r * conj(d(target_r)) )
            // # V(d(err_r)) = 2 * M_Im * M_right(measure_r * source_r) * M_Conj * V(d(target_r))
            // # d(err_r) / d(target_t) = 2 * M_Im * M_right(measure_r * source_r) * M_Conj
            Matx34d drdtq = 2.0 * (m_right(rotMeasured * sourceQuat) * M_Conj).get_minor<3, 4>(1, 0);

            // d(err_t) = d(target_t - measure_r * source_t * conj(measure_r) - measure_t) = <measure_* are constants> =
            // d(target_t) - measure_r * d(source_t) * conj(measure_r)
            Matx34d dtdsq, dtdtq;
            // source_t is rotated by measure_r so its jacobian is just rotation matrix of measure_r
            Matx33d dtdst = - rotMeasured.toRotMat3x3(QUAT_ASSUME_UNIT);
            Matx33d dtdtt = Matx33d::eye();

            Matx33d z;
            sqj = concatVert(dtdsq, drdsq);
            tqj = concatVert(dtdtq, drdtq);
            stj = concatVert(dtdst, z);
            ttj = concatVert(dtdtt, z);

            stj = sqrtInfoMatrix * stj;
            ttj = sqrtInfoMatrix * ttj;
            sqj = sqrtInfoMatrix * sqj;
            tqj = sqrtInfoMatrix * tqj;
        }

        return res.ddot(res);
    }
    else
    {

        // err_r = 2*Im(conj(rel_r) * measure_r) = 2*Im(conj(target_r) * source_r * measure_r)
        // err_t = conj(source_r) * (target_t - source_t) * source_r - measure_t

        Quatd sourceQuatInv = sourceQuat.conjugate();
        Vec3d deltaTrans = targetTrans - sourceTrans;

        Quatd relativeQuat = sourceQuatInv * targetQuat;
        Vec3d relativeTrans = sourceQuatInv.toRotMat3x3(cv::QUAT_ASSUME_UNIT) * deltaTrans;

        //! Definition should actually be relativeQuat * rotMeasured.conjugate()
        Quatd deltaRot = relativeQuat.conjugate() * rotMeasured;

        Vec3d terr = relativeTrans - transMeasured;
        Vec3d rerr = 2.0 * Vec3d(deltaRot.x, deltaRot.y, deltaRot.z);
        Vec6d rterr(terr[0], terr[1], terr[2], rerr[0], rerr[1], rerr[2]);

        res = sqrtInfoMatrix * rterr;

        if (needJacobians)
        {
            // d(err_r) = 2*Im(d(conj(target_r) * source_r * measure_r)) = < measure_r is constant > =
            // 2*Im((conj(d(target_r)) * source_r + conj(target_r) * d(source_r)) * measure_r)
            // d(target_r) == 0:
            //  # d(err_r) = 2*Im(conj(target_r) * d(source_r) * measure_r)
            //  # V(d(err_r)) = 2 * M_Im * M_right(measure_r) * M_left(conj(target_r)) * V(d(source_r))
            //  # d(err_r) / d(source_r) = 2 * M_Im * M_right(measure_r) * M_left(conj(target_r))
            Matx34d drdsq = 2.0 * (m_right(rotMeasured) * m_left(targetQuat.conjugate())).get_minor<3, 4>(1, 0);

            // d(source_r) == 0:
            //  # d(err_r) = 2*Im(conj(d(target_r)) * source_r * measure_r)
            //  # V(d(err_r)) = 2 * M_Im * M_right(source_r * measure_r) * M_Conj * V(d(target_r))
            //  # d(err_r) / d(target_r) = 2 * M_Im * M_right(source_r * measure_r) * M_Conj
            Matx34d drdtq = 2.0 * (m_right(sourceQuat * rotMeasured) * M_Conj).get_minor<3, 4>(1, 0);

            // d(err_t) = d(conj(source_r) * (target_t - source_t) * source_r) =
            // conj(source_r) * (d(target_t) - d(source_t)) * source_r +
            // conj(d(source_r)) * (target_t - source_t) * source_r +
            // conj(source_r) * (target_t - source_t) * d(source_r) =
            // <conj(a*b) == conj(b)*conj(a), conj(target_t - source_t) = - (target_t - source_t), 2 * Im(x) = (x - conj(x))>
            // conj(source_r) * (d(target_t) - d(source_t)) * source_r +
            // 2 * Im(conj(source_r) * (target_t - source_t) * d(source_r))
            // d(*_t) == 0:
            //  # d(err_t) = 2 * Im(conj(source_r) * (target_t - source_t) * d(source_r))
            //  # V(d(err_t)) = 2 * M_Im * M_left(conj(source_r) * (target_t - source_t)) * V(d(source_r))
            //  # d(err_t) / d(source_r) = 2 * M_Im * M_left(conj(source_r) * (target_t - source_t))
            Matx34d dtdsq = 2 * m_left(sourceQuatInv * Quatd(0, deltaTrans[0], deltaTrans[1], deltaTrans[2])).get_minor<3, 4>(1, 0);
            // deltaTrans is rotated by sourceQuatInv, so the jacobian is rot matrix of sourceQuatInv by +1 or -1
            Matx33d dtdtt = sourceQuatInv.toRotMat3x3(QUAT_ASSUME_UNIT);
            Matx33d dtdst = -dtdtt;

            Matx33d z;
            sqj = concatVert(dtdsq, drdsq);
            tqj = concatVert(Matx34d(), drdtq);
            stj = concatVert(dtdst, z);
            ttj = concatVert(dtdtt, z);

            stj = sqrtInfoMatrix * stj;
            ttj = sqrtInfoMatrix * ttj;
            sqj = sqrtInfoMatrix * sqj;
            tqj = sqrtInfoMatrix * tqj;
        }

        return res.ddot(res);
    }
}


double PoseGraphImpl::calcEnergy() const
{
    return calcEnergyNodes(nodes);
}


// estimate current energy
double PoseGraphImpl::calcEnergyNodes(const std::map<size_t, Node>& newNodes) const
{
    double totalErr = 0;
    for (const auto& e : edges)
    {
        Pose3d srcP = newNodes.at(e.sourceNodeId).pose;
        Pose3d tgtP = newNodes.at(e.targetNodeId).pose;

        Vec6d res;
        Matx<double, 6, 3> stj, ttj;
        Matx<double, 6, 4> sqj, tqj;
        double err = poseError(srcP.q, srcP.t, tgtP.q, tgtP.t, e.pose.q, e.pose.t, e.sqrtInfo,
                               (errorApplyFlags == ERROR_LEFT), /* needJacobians = */ false, sqj, stj, tqj, ttj, res);

        totalErr += err;
    }
    return totalErr * 0.5;
}


// J := J * d_inv, d_inv = make_diag(di)
// J^T*J := (J * d_inv)^T * J * d_inv = diag(di) * (J^T * J) * diag(di) = eltwise_mul(J^T*J, di*di^T)
// J^T*b := (J * d_inv)^T * b = d_inv^T * J^T*b = eltwise_mul(J^T*b, di)
static void doJacobiScalingSparse(BlockSparseMat<double, 6, 6>& jtj, Mat_<double>& jtb, const Mat_<double>& di)
{
    // scaling J^T*J
    for (auto& ijv : jtj.ijValue)
    {
        Point2i bpt = ijv.first;
        Matx66d& m = ijv.second;
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                Point2i pt(bpt.x * 6 + i, bpt.y * 6 + j);
                m(i, j) *= di(pt.x) * di(pt.y);
            }
        }
    }

    // scaling J^T*b
    jtb = jtb.mul(di);
}


class PoseGraphLevMarqBackend : public LevMarqBackend
{
public:
    PoseGraphLevMarqBackend(PoseGraphImpl* pg_) :
        LevMarqBackend(),
        pg(pg_),
        jtj(0),
        jtb(),
        tempNodes(),
        useGeo(),
        geoNodes(),
        jtrvv(),
        jtCached(),
        decomposition(),
        numNodes(),
        numEdges(),
        placesIds(),
        idToPlace(),
        robustWeights(),
        nVarNodes()
    {
        if (!pg->isValid())
        {
            CV_Error(cv::Error::Code::StsBadArg, "Invalid PoseGraph that is either not connected or has invalid nodes");
        }

        this->numNodes = pg->getNumNodes();
        this->numEdges = pg->getNumEdges();

        // Allocate indices for nodes
        for (const auto& ni : pg->nodes)
        {
            if (!ni.second.isFixed)
            {
                this->idToPlace[ni.first] = this->placesIds.size();
                this->placesIds.push_back(ni.first);
            }
        }

        this->nVarNodes = this->placesIds.size();
        if (!this->nVarNodes)
        {
            CV_Error(cv::Error::Code::StsBadArg, "PoseGraph contains no non-constant nodes, skipping optimization");
        }

        if (!this->numEdges)
        {
            CV_Error(cv::Error::Code::StsBadArg, "PoseGraph has no edges, no optimization to be done");
        }

        robustFlags = pg->robustFlags;


        CV_LOG_INFO(NULL, "Optimizing PoseGraph with " << this->numNodes << " nodes and " << this->numEdges << " edges");

        this->nVars = this->nVarNodes * 6;
    }


    virtual bool calcFunc(double& energy, bool calcEnergy = true, bool calcJacobian = false) CV_OVERRIDE
    {
        std::map<size_t, PoseGraphImpl::Node>& nodes = tempNodes;
        bool useLeft = (pg->errorApplyFlags & ERROR_LEFT);

        std::vector<cv::Matx<double, 7, 6>> cachedJac;
        if (calcJacobian)
        {
            jtj.clear();
            std::fill(jtb.begin(), jtb.end(), 0.0);

            // caching nodes jacobians
            for (auto id : placesIds)
            {
                cachedJac.push_back(nodes.at(id).pose.expJacobian());
            }

            if (useGeo)
                jtCached.clear();
        }

        bool enableRobust = (robustFlags != ROBUST_DISABLED) && numEdges >= 10;

        // Robust weights should be calculated together with jacobian calculation only
        if (enableRobust && calcJacobian)
        {
            std::vector<double> errors;
            std::vector<size_t> idxs;
            int ei = 0;
            for (const auto& e : pg->edges)
            {
                size_t srcId = e.sourceNodeId, dstId = e.targetNodeId;
                const PoseGraphImpl::Node& srcNode = nodes.at(srcId);
                const PoseGraphImpl::Node& dstNode = nodes.at(dstId);

                const PoseGraphImpl::Pose3d& srcP = srcNode.pose;
                const PoseGraphImpl::Pose3d& tgtP = dstNode.pose;
                bool srcFixed = srcNode.isFixed;
                bool dstFixed = dstNode.isFixed;

                // fixed edges have fixed weight == 1.0
                if (!(srcFixed && dstFixed))
                {
                    Vec6d res;
                    Matx<double, 6, 3> stj, ttj;
                    Matx<double, 6, 4> sqj, tqj;

                    double err = poseError(srcP.q, srcP.t, tgtP.q, tgtP.t, e.pose.q, e.pose.t, e.sqrtInfo,
                                           useLeft, /* needJacobians = */ false, sqj, stj, tqj, ttj, res);

                    errors.push_back(err);
                    idxs.push_back(ei);
                }
                ei++;
            }

            double mean = 0;
            double sigma = (robustFlags & ROBUST_MAD) ? madEstimate(errors, mean) :
                           (robustFlags & ROBUST_STD) ? stdEstimate(errors, mean) : 0.0;

            const double EPS = 1e-5;
            for (int i = 0; i < errors.size(); i++)
            {
                double rw = 0;
                // special case when errors are mostly the same
                // just turn off others
                if (sigma < EPS)
                {
                    rw = abs(errors[i] - mean) < EPS ? 1.0 : 0.0;
                }
                else
                {
                    rw = (robustFlags & ROBUST_TUKEY) ? tukeyWeightSq(errors[i], sigma) :
                         (robustFlags & ROBUST_HUBER) ? huberWeightSq(errors[i], sigma) : 0.0;
                }
                robustWeights[idxs[i]] = rw;
            }
        }

        int ei = 0;
        double totalErr = 0.0;
        for (const auto& e : pg->edges)
        {
            size_t srcId = e.sourceNodeId, dstId = e.targetNodeId;
            const PoseGraphImpl::Node& srcNode = nodes.at(srcId);
            const PoseGraphImpl::Node& dstNode = nodes.at(dstId);

            const PoseGraphImpl::Pose3d& srcP = srcNode.pose;
            const PoseGraphImpl::Pose3d& tgtP = dstNode.pose;
            bool srcFixed = srcNode.isFixed;
            bool dstFixed = dstNode.isFixed;

            Vec6d res;
            Matx<double, 6, 3> stj, ttj;
            Matx<double, 6, 4> sqj, tqj;

            bool edgeIsFixed = srcFixed && dstFixed;
            double weight = (!enableRobust || edgeIsFixed) ? 1.0 : robustWeights[ei] ;

            double err = poseError(srcP.q, srcP.t, tgtP.q, tgtP.t, e.pose.q, e.pose.t, e.sqrtInfo,
                                   useLeft, /* needJacobians = */ calcJacobian, sqj, stj, tqj, ttj, res);
            totalErr += weight * err;

            if (calcJacobian)
            {
                size_t srcPlace = (size_t)(-1), dstPlace = (size_t)(-1);
                Matx66d sj, tj;
                if (!srcFixed)
                {
                    srcPlace = idToPlace.at(srcId);
                    sj = concatHor(sqj, stj) * cachedJac[srcPlace];

                    jtj.refBlock(srcPlace, srcPlace) += weight * sj.t() * sj;

                    Vec6d jtbSrc = weight * sj.t() * res;
                    for (int i = 0; i < 6; i++)
                    {
                        jtb(6 * (int)srcPlace + i) += jtbSrc[i];
                    }
                }

                if (!dstFixed)
                {
                    dstPlace = idToPlace.at(dstId);
                    tj = concatHor(tqj, ttj) * cachedJac[dstPlace];

                    jtj.refBlock(dstPlace, dstPlace) += weight * tj.t() * tj;

                    Vec6d jtbDst = weight * tj.t() * res;
                    for (int i = 0; i < 6; i++)
                    {
                        jtb(6 * (int)dstPlace + i) += jtbDst[i];
                    }
                }

                if (!(srcFixed || dstFixed))
                {
                    Matx66d sjttj = weight * sj.t() * tj;
                    jtj.refBlock(srcPlace, dstPlace) += sjttj;
                    jtj.refBlock(dstPlace, srcPlace) += sjttj.t();
                }

                if (useGeo)
                {
                    jtCached.push_back({ sj, tj });
                }
            }

            ei++;
        }

        if (calcEnergy)
        {
            energy = totalErr * 0.5;
        }

        return true;
    }

    virtual bool enableGeo() CV_OVERRIDE
    {
        useGeo = true;
        return true;
    }

    // adds d to current variables and writes result to probe vars or geo vars
    virtual void currentOplusX(const Mat_<double>& d, bool geo = false) CV_OVERRIDE
    {
        if (geo && !useGeo)
        {
            CV_Error(CV_StsBadArg, "Geodesic acceleration is disabled");
        }

        std::map<size_t, PoseGraphImpl::Node>& nodes = geo ? geoNodes : tempNodes;

        nodes = pg->nodes;

        for (size_t i = 0; i < nVarNodes; i++)
        {
            Vec6d dx(d[0] + (i * 6));
            PoseGraphImpl::Pose3d& p = nodes.at(placesIds[i]).pose;

            p = p.oplus(dx);
        }
    }

    virtual void prepareVars() CV_OVERRIDE
    {
        jtj = BlockSparseMat<double, 6, 6>(nVarNodes);
        jtb = Mat_<double>((int)nVars, 1);
        tempNodes = pg->nodes;
        if (useGeo)
            geoNodes = pg->nodes;
        if (robustFlags != PoseGraphRobustFlags::ROBUST_DISABLED)
        {
            robustWeights.resize(pg->edges.size());
        }
    }

    // decomposes LevMarq matrix before solution
    virtual bool decompose() CV_OVERRIDE
    {
        //DEBUG
        //return jtj.decompose(decomposition, false);
        bool res = jtj.decompose(decomposition, false);
        if (!res)
        {
            //SIC! 0.0001 is zero threshold
            Mat_<double> toPrint(int(jtj.nBlocks * 6), int(jtj.nBlocks * 6));
            for (const auto& ijv : jtj.ijValue)
            {
                int xb = ijv.first.x, yb = ijv.first.y;
                Matx66d vblock = ijv.second;
                for (size_t i = 0; i < 6; i++)
                {
                    for (size_t j = 0; j < 6; j++)
                    {
                        double val = vblock((int)i, (int)j);
                        toPrint(6 * yb + j, 6 * xb + i) = val;
                    }
                }
            }
            std::cout << "placesIds:" << std::endl;
            for (auto v : placesIds)
            {
                std::cout << " " << v;
            }
            std::cout << std::endl;
            std::cout << "jtj:" << std::endl;
            std::cout << toPrint << std::endl;
            std::cout << std::endl;
        }
        return res;
    }

    // solves LevMarq equation (J^T*J + lmdiag) * x = -right for current iteration using existing decomposition
    // right can be equal to J^T*b for LevMarq equation or J^T*rvv for geodesic acceleration equation
    virtual bool solveDecomposed(const Mat_<double>& right, Mat_<double>& x) CV_OVERRIDE
    {
        return jtj.solveDecomposed(decomposition, -right, x);
    }

    // calculates J^T*f(geo)
    virtual bool calcJtbv(Mat_<double>& jtbv) CV_OVERRIDE
    {
        jtbv.setZero();

        bool useLeft = (pg->errorApplyFlags & ERROR_LEFT);
        int ei = 0;
        for (const auto& e : pg->edges)
        {
            size_t srcId = e.sourceNodeId, dstId = e.targetNodeId;
            const PoseGraphImpl::Node& srcNode = geoNodes.at(srcId);
            const PoseGraphImpl::Node& dstNode = geoNodes.at(dstId);

            const PoseGraphImpl::Pose3d& srcP = srcNode.pose;
            const PoseGraphImpl::Pose3d& tgtP = dstNode.pose;
            bool srcFixed = srcNode.isFixed;
            bool dstFixed = dstNode.isFixed;

            Vec6d res;
            // dummy vars
            Matx<double, 6, 3> stj, ttj;
            Matx<double, 6, 4> sqj, tqj;

            poseError(srcP.q, srcP.t, tgtP.q, tgtP.t, e.pose.q, e.pose.t, e.sqrtInfo,
                      useLeft, /* needJacobians = */ false, sqj, stj, tqj, ttj, res);

            // jtCached and robustWeights should be already calculated by calcFunc() at this point
            size_t srcPlace = (size_t)(-1), dstPlace = (size_t)(-1);
            Matx66d sj = jtCached[ei].first, tj = jtCached[ei].second;

            double weight = (robustFlags != ROBUST_DISABLED) ? robustWeights[ei] : 1.0;

            if (!srcFixed)
            {
                srcPlace = idToPlace.at(srcId);

                Vec6d jtbSrc = sj.t() * res;
                for (int i = 0; i < 6; i++)
                {
                    jtbv(6 * (int)srcPlace + i) += weight * jtbSrc[i];
                }
            }

            if (!dstFixed)
            {
                dstPlace = idToPlace.at(dstId);

                Vec6d jtbDst = tj.t() * res;
                for (int i = 0; i < 6; i++)
                {
                    jtbv(6 * (int)dstPlace + i) += weight * jtbDst[i];
                }
            }

            ei++;
        }

        return true;
    }

    virtual const Mat_<double> getDiag() CV_OVERRIDE
    {
        return jtj.diagonal();
    }

    virtual const Mat_<double> getJtb() CV_OVERRIDE
    {
        return jtb;
    }

    virtual void setDiag(const Mat_<double>& d) CV_OVERRIDE
    {
        for (size_t i = 0; i < nVars; i++)
        {
            jtj.refElem(i, i) = d((int)i);
        }
    }

    virtual void doJacobiScaling(const Mat_<double>& di) CV_OVERRIDE
    {
        doJacobiScalingSparse(jtj, jtb, di);
    }


    virtual void acceptProbe() CV_OVERRIDE
    {
        pg->nodes = tempNodes;
    }

    PoseGraphImpl* pg;

    // J^T*J matrix
    BlockSparseMat<double, 6, 6> jtj;
    // J^T*b vector
    Mat_<double> jtb;

    // Probe variable for different lambda tryout
    std::map<size_t, PoseGraphImpl::Node> tempNodes;

    // For geodesic acceleration
    bool useGeo;
    std::map<size_t, PoseGraphImpl::Node> geoNodes;
    Mat_<double> jtrvv;
    std::vector<std::pair<Matx66d, Matx66d>> jtCached;

    // Used for keeping intermediate matrix decomposition for further linear solve operations
    BlockSparseMat<double, 6, 6>::Decomposition decomposition;

    // The rest members are generated from pg
    size_t nVars;
    size_t numNodes;
    size_t numEdges;

    // Structures to convert node id to place in variables vector and back
    std::vector<size_t> placesIds;
    std::map<size_t, size_t> idToPlace;

    // Used for noise filtering
    int robustFlags;
    std::vector<double> robustWeights;

    size_t nVarNodes;
};


LevMarq::Report PoseGraphImpl::optimize()
{
    if (!lm)
        createOptimizer();
    return lm->optimize();
}


Ptr<PoseGraph> PoseGraph::create(int robustFlags, int errorApplyFlags)
{
    return makePtr<PoseGraphImpl>(robustFlags, errorApplyFlags);
}

#else

Ptr<PoseGraph> PoseGraph::create(int, int)
{
    CV_Error(Error::StsNotImplemented, "Eigen library required for sparse matrix solve during pose graph optimization, dense solver is not implemented");
}

#endif

PoseGraph::~PoseGraph() { }

}  // namespace detail
}  // namespace cv
