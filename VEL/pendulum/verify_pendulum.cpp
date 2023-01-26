#include "../flowstar/flowstar-toolbox/Continuous.h"

using namespace std;
using namespace flowstar;

double relu(double x) {
	return max(x, 0.0);
}

double reach_loss(vector<double> reach_set, vector<double> target_set) {
	double val = 0;
	val = max(relu(reach_set[1] - target_set[1]), val); // x_xup
	val = max(relu(target_set[0] - reach_set[0]), val); // x_inf

	val = max(relu(reach_set[3] - target_set[3]), val); // y_sup
	val = max(relu(target_set[2] - reach_set[2]), val); // y_inf
	
	return val;
}

int main(int argc, char *argv[])
{
    
//    intervalNumPrecision = 300;
    
	// Declaration of the state variables.
	unsigned int numVars = 3;
    
    Variables vars;

	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int u_id = vars.declareVar("u");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
    ODE<Real> dynamics({"x2", "15 * sin(x1) + 3*u", "0"}, vars);
    
	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 7;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.005, order);

	// time horizon for a single control step
//	setting.setTime(0.1);

	// cutoff threshold
	setting.setCutoffThreshold(1e-10);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

    //setting.g_setting.prepareForReachability(15);
  //  cout << "--------" << setting.g_setting.factorial_rec.size() << ", " << setting.g_setting.power_4.size() << ", " << setting.g_setting.double_factorial.size() << endl;

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	Interval init_x1(-0.1, 0.1), init_x2(-0.1, 0.1);
	// Interval init_x0(-0.25 - w, -0.25 + w), init_x1(-0.25 - w, -0.25 + w), init_x2(0.35 - w, 0.35 + w), init_x3(-0.35 - w, -0.35 + w), init_x4(0.45 - w, 0.45 + w), init_x5(-0.35 - w, -0.35 + w);
	Interval init_u(0);
	std::vector<Interval> X0;
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_u);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);
    
    Symbolic_Remainder symbolic_remainder(initial_set, 2000);

	// no unsafe set
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	double err_max = 0;
	time_t start_timer;
	time_t end_timer;
	double seconds;
	time(&start_timer);

	string s1 = argv[1];
	string s2 = argv[2];
	string s3 = argv[3];
	string s4 = argv[4];
	bool plot = false;
	if (argc == 6) {
		string plt = argv[5];
		if (plt == "--plot") {
			plot = true;
		}
	}
	double score = 0;

	for (int iter = 0; iter < 15; ++iter)
	{
		// cout << "Step " << iter << " starts.      " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		
		
		string strExpU = "cos(x1)*(" + s1 + ") + sin(x1)*(" + s2 + ") + x2 *(" + s3 + ") + (" + s4 + ")";
        // cout << strExpU << endl;
		Expression<Real> exp_u(strExpU, vars);

        TaylorModel<Real> tm_u;
        exp_u.evaluate(tm_u, initial_set.tmvPre.tms, order, initial_set.domain, setting.tm_setting.cutoff_threshold, setting.g_setting);
        tm_u.remainder.bloat(0);

        initial_set.tmvPre.tms[u_id] = tm_u;

        dynamics.reach(result, initial_set, 0.05, setting, safeSet);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
			vector<Interval> inter_box;
			string safe_result;
			result.fp_end_of_time.intEval(inter_box, order, setting.tm_setting.cutoff_threshold);
            // for (int i = 0; i < 3; ++i) {
            //     cout << inter_box[i].inf() << " ";
            //     cout << inter_box[i].sup() << " ";
            // }
            // cout << endl;
			if (inter_box[0].inf() >= -0.1 && inter_box[0].sup() <= 0.1 && inter_box[1].inf() >= -0.1 && inter_box[1].sup() <= 0.1) {
				cout << "returned to initial region, loss is 0" << endl;
                break;
			}
			if (inter_box[0].inf() <= -1.5708 || inter_box[0].sup() >= 1.5708 || inter_box[1].inf() <= -1.5708 || inter_box[1].sup() >= 1.5708) {
				cout << "out the safe regoin" << endl;
                break;
			}
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			break;
		}
	}

	if (plot) {
		// plot the flowpipes in the x-y plane
		result.transformToTaylorModels(setting);

		Plot_Setting plot_setting(vars);
		int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
		if (mkres < 0 && errno != EEXIST)
		{
			printf("Can not create the directory for images.\n");
			exit(1);
		}
		// you need to create a subdir named outputs
		// the file name is example.m and it is put in the subdir outputs
		plot_setting.setOutputDims("x1", "x2");
		plot_setting.plot_2D_interval_MATLAB("./outputs/", "pendulum", result.tmv_flowpipes, setting);
	}
	return 0;
}