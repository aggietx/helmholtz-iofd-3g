#pragma once

#include "common.hpp"

namespace stolk {

using ApplyFn = std::function<void(const Vector&, Vector&)>;
using PrecondFn = std::function<void(const Vector&, Vector&)>;

struct GmresResult {
  bool converged = false;
  int iterations = 0;
  int cycles = 0;
  Real initial_residual = 0.0;
  Real final_residual = 0.0;
  Real final_predicted_iterations = 0.0;
  std::string stop_reason;
  std::vector<Real> relative_history;
  std::vector<Real> elapsed_history;
  std::vector<int> outer_iteration_history;
};

void jacobi_apply(const Vector& rhs, Vector& out, const Vector& diag,
                  Real omega, int sweeps, const ApplyFn& apply);

void fixed_fgmres_steps(const ApplyFn& apply, const Vector& rhs, Vector& x,
                        int steps, const PrecondFn& precond);

GmresResult fgmres_solve(const ApplyFn& apply, const Vector& rhs, Vector& x,
                         int restart, int max_cycles, Real tol,
                         const PrecondFn& precond, bool verbose);

GmresResult gmres_solve_fixed_precond(const ApplyFn& apply, const Vector& rhs,
                                      Vector& x, int restart, int max_cycles,
                                      Real tol, const PrecondFn& precond,
                                      bool verbose);

}  // namespace stolk
