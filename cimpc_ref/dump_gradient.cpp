// CI-MPC (Kim et al., IJRR 2025) 의 C++ 구현에서 접촉 impulse 와 그 gradient 를
// 덤프한다. DynaFlow 쪽 JAX 구현(cimpc_grad.cimpc_jacobian)과 대조하기 위한 것.
//
// 입력  : 상태 파일 (q 19개, v 18개, tau 18개, rho 1개; 공백/줄바꿈 구분)
//          q 는 Pinocchio 규약 [x y z qx qy qz qw joint*12]  <- MuJoCo 는 qw 가 앞
// 출력  : 이름 붙은 블록들을 stdout 으로. Python 에서 파싱한다.
#include "contactSolverGradient.h"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include <fstream>
#include <iomanip>

static void dumpMat(const std::string& name, const Eigen::MatrixXd& M) {
    std::cout << "### " << name << " " << M.rows() << " " << M.cols() << "\n"
              << std::setprecision(17) << M << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: dump_gradient <state_file>\n"; return 1; }

    const std::string base = "/home/andyjeong/workspace/general_contact_TO/contact_solving_gradient/rsc/";
    pin::Model model, model_hessian;
    pin::JointModelFreeFlyer rj, rjh;
    pin::urdf::buildModel(base + "Hound.urdf", rj, model);
    pin::urdf::buildModel(base + "Hound_gradient.urdf", rjh, model_hessian);
    pin::Data data(model);

    // ---- 상태 읽기 ----
    Matrix<double, nq, 1> q;  Matrix<double, nv, 1> v, tau;  double rho;
    { std::ifstream f(argv[1]);
      for (int i = 0; i < nq; ++i) f >> q(i);
      for (int i = 0; i < nv; ++i) f >> v(i);
      for (int i = 0; i < nv; ++i) f >> tau(i);
      f >> rho; }
    q.segment(3, 4) = q.segment(3, 4).normalized();

    // 선택 인자: tangent 방향 섭동 (dA/dq, db/dq 를 수치로 얻기 위함)
    //   argv[2] = 섭동할 tangent 인덱스 (0..nv-1), argv[3] = 크기
    // pin::integrate 로 manifold 위에서 섭동해야 quaternion 이 깨지지 않는다.
    if (argc >= 4) {
        int idx = std::atoi(argv[2]); double eps = std::atof(argv[3]);
        Matrix<double, nv, 1> dv_pert; dv_pert.setZero(); dv_pert(idx) = eps;
        Matrix<double, nq, 1> q_out;
        pin::integrate(model, q, dv_pert, q_out);
        q = q_out;
    }

    // ---- 접촉 impulse 풀기 ----
    Matrix<double, dim, fnum> lambda;
    Matrix<int, fnum, 1> cstate, sstate;
    Eigen::VectorXd xnext(nq + nv);

    ContactSolverGradient CM(model, model_hessian);
    CM.SetState(q, v, tau);
    CM.SolveContactImpulseInitialize();
    CM.SolveContactImpulse(lambda, cstate, sstate, xnext);

    // ---- gradient ----
    Matrix<double, nq + 4, 1> qh;  Matrix<double, nv + 4, 1> vh;
    MappingToHessianModelnq(q, qh);
    MappingToHessianModelnv(v, vh);

    const int nrow = cstate.sum() * 3 - sstate.sum() * 2;   // clamping 3행, sliding 1행
    Eigen::MatrixXd dlam(nrow, nv * 3);                     // 열: [dq | dv | dtau]
    CM.SetStateHessian(qh, vh);
    CM.GetGradientContactImpulseInitialize(lambda, rho, cstate, sstate);
    CM.GetGradient(dlam);

    // ---- 모델 동등성 확인용 (질량행렬, 발 자코비안, 발 위치) ----
    pin::crba(model, data, q);
    Eigen::MatrixXd M = data.M;
    M.triangularView<Eigen::StrictlyLower>() = M.transpose().triangularView<Eigen::StrictlyLower>();

    pin::computeJointJacobians(model, data, q);
    pin::framesForwardKinematics(model, data, q);
    Eigen::MatrixXd J(3 * fnum, nv), fpos(3, fnum);
    Matrix<int, fnum, 1> FrameId; FrameId << 9, 17, 25, 33;   // FL, FR, RL, RR
    for (int k = 0; k < fnum; ++k) {
        Eigen::Matrix<double, 6, nv> J6; J6.setZero();
        pin::getFrameJacobian(model, data, FrameId(k), pin::LOCAL_WORLD_ALIGNED, J6);
        J.middleRows(3 * k, 3) = J6.topRows(3);
        fpos.col(k) = data.oMf[FrameId(k)].translation();
    }

    std::cout << "### meta 1 5\n" << dt << " " << mu << " " << rho << " "
              << cstate.sum() << " " << sstate.sum() << "\n";
    dumpMat("contact_state", cstate.cast<double>());
    dumpMat("sliding_state", sstate.cast<double>());
    dumpMat("lambda_lw", lambda);          // (3, 4) 발별 impulse, local-world-aligned
    dumpMat("foot_pos", fpos);
    dumpMat("mass_matrix", M);
    dumpMat("foot_jacobian", J);           // (12, 18)
    dumpMat("dlambda_dz", dlam);           // (nrow, 54)
    Eigen::VectorXd nle = pin::nonLinearEffects(model, data, q, v);
    dumpMat("nle", nle);              // M a + nle = tau  (Pinocchio 규약)
    dumpMat("q_used", q);
    dumpMat("xnext", xnext);
    return 0;
}
