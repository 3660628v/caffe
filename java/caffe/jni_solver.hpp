#ifndef CAFFE_JNI_SOLVER_HPP_
#define CAFFE_JNI_SOLVER_HPP_

#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"

using boost::dynamic_pointer_cast;
using boost::shared_ptr;
using caffe::DataLayer;
using caffe::Layer;
using caffe::NetParameter;
using caffe::Solver;
using caffe::SolverParameter;
using caffe::vector;

/**
 * The JNI wrapper for caffe solver. Marked as friend class of solver to access
 * some protected fields. All public functions correspond to JNI interface.
 *
 * Use this class to make minimal changes to caffe's solver class while
 * providing options to choose from different float precisions.
 */
template <typename Dtype>
class JNISolver {
 public:
  explicit JNISolver(const SolverParameter& solver_param) {
    solver = caffe::GetSolver<Dtype>(solver_param);
    // share weights of all test layers with trainig layers
    for (int i = 0; i < solver->test_nets().size(); ++i) {
      solver->test_nets()[i]->ShareTrainedLayersWith(solver->net().get());
    }
  }
  ~JNISolver() { delete solver; }

  inline Dtype train(int iteration, bool update_diff)  {
    return solver->Step(iteration, true, update_diff);
  }

  inline Dtype test(int iteration) {
    return solver->Test();
  }

  inline void setIteration(int iteration) {
    solver->iter_ = iteration;
    if (solver->param_.lr_policy() == "multistep") {
      solver->current_step_ = 0;
      while (solver->current_step_ < solver->param_.stepvalue_size() &&
        solver->iter_ >= solver->param_.stepvalue(solver->current_step_)) {
        solver->current_step_++;
      }
    }
  }

  inline void setBatchSize(int batch_size) {
    const vector<shared_ptr<Layer<Dtype> > >& layers = solver->net_->layers();
    for (int i = 0; i < layers.size(); i++) {
      shared_ptr<DataLayer<Dtype> > layer =
        dynamic_pointer_cast<DataLayer<Dtype> >(layers[i]);
      // only set batch size on DataLayer
      if (layer) layer->set_batch_size(batch_size);
    }
  }

  inline void updateParameter(const SolverParameter& param) {
    solver->param_.set_momentum(param.momentum());
    solver->param_.set_weight_decay(param.weight_decay());
    solver->param_.set_iter_size(param.iter_size());
    // Learning Rate related fields
    solver->param_.set_base_lr(param.base_lr());
    solver->param_.set_lr_policy(param.lr_policy());
    solver->param_.set_gamma(param.gamma());
    solver->param_.set_power(param.power());
    solver->param_.set_stepsize(param.stepsize());
    solver->param_.mutable_stepvalue()->CopyFrom(param.stepvalue());
    // For AdaGrad Only
    solver->param_.set_delta(param.delta());
    // For RMSProp Only
    solver->param_.set_rms_decay(param.rms_decay());
  }

  inline void getWeight(NetParameter* weight, bool diff) {
    solver->net_->ToProto(weight, diff);
  }

  inline void setWeight(const NetParameter& weight) {
    solver->net_->CopyTrainedLayersFrom(weight);
  }

  inline void mergeDelta(const NetParameter& delta, NetParameter* weight) {
    // Set delta
    solver->net()->CopyTrainedLayersFrom(delta);
    // Update weights
    solver->Step(1, false, true);
    // Get weights
    solver->net()->ToProto(weight);
  }

 protected:
  Solver<Dtype>* solver;

  DISABLE_COPY_AND_ASSIGN(JNISolver);
};

#endif  // CAFFE_JNI_SOLVER_HPP_
