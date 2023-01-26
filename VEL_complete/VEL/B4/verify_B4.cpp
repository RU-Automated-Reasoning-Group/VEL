#include "../flowstar/flowstar-toolbox/Continuous.h"

using namespace std;
using namespace flowstar;

double relu(double x) {
	return max(x, 0.0);
}

double reach_loss(vector<double> reach_set, vector<double> target_set) {
    // loss only uses the first 2 dimension in this benchmark
	double val = 0;
	val = max(relu(reach_set[1] - target_set[1]), val); // x_xup
	val = max(relu(target_set[0] - reach_set[0]), val); // x_inf

	val = max(relu(reach_set[3] - target_set[3]), val); // y_sup
	val = max(relu(target_set[2] - reach_set[2]), val); // y_inf
	
	// val = max(relu(reach_set[5] - target_set[5]), val); // z_sup
	// val = max(relu(target_set[4] - reach_set[4]), val); // z_inf

	return val;
}

int main(int argc, char *argv[])
{
    
//    intervalNumPrecision = 300;
    
	// Declaration of the state variables.
	unsigned int numVars = 4;
    
    Variables vars;

	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int u_id = vars.declareVar("u");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
    ODE<Real> dynamics({"-x1 + x2 - x3", "-x1 * (x3 + 1) - x2", "-x1 + u", "0"}, vars);
    
	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 8;

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
	Interval init_x1(0.25, 0.27), init_x2(0.08, 0.1), init_x3(0.25, 0.27);
	// Interval init_x0(-0.25 - w, -0.25 + w), init_x1(-0.25 - w, -0.25 + w), init_x2(0.35 - w, 0.35 + w), init_x3(-0.35 - w, -0.35 + w), init_x4(0.45 - w, 0.45 + w), init_x5(-0.35 - w, -0.35 + w);
	Interval init_u(0);
	std::vector<Interval> X0;
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
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
	// perform 35 control steps
	for (int iter = 0; iter < 5; ++iter)
	{
		// cout << "Step " << iter << " starts.      " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		
		
		string strExpU = "x1*(" + s1 + ") + x2*(" + s2 + ") + x3 * (" + s3 + ") + (" + s4 + ")";
        // cout << strExpU << endl;
		Expression<Real> exp_u(strExpU, vars);

        TaylorModel<Real> tm_u;
        exp_u.evaluate(tm_u, initial_set.tmvPre.tms, order, initial_set.domain, setting.tm_setting.cutoff_threshold, setting.g_setting);
        tm_u.remainder.bloat(0);

        initial_set.tmvPre.tms[u_id] = tm_u;

        dynamics.reach(result, initial_set, 0.1, setting, safeSet, symbolic_remainder);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
            vector<Interval> inter_box;
			result.fp_end_of_time.intEval(inter_box, order, setting.tm_setting.cutoff_threshold);
			// for (int i = 0; i < 4; ++i) {
            //     cout << inter_box[i].inf() << " ";
            //     cout << inter_box[i].sup() << " ";
            // }
            // cout << endl;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			break;
		}
	}

	vector<double> target_set;
	target_set.push_back(-0.05);
	target_set.push_back(0.05);
	target_set.push_back(-0.05);
	target_set.push_back(0.0);

	vector<Interval> end_box;
	result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);
	vector<double> reach_set;
	reach_set.push_back(end_box[0].inf());
	reach_set.push_back(end_box[0].sup());
	reach_set.push_back(end_box[1].inf());
	reach_set.push_back(end_box[1].sup());

	double r_loss = reach_loss(reach_set, target_set);
	score += r_loss;
	time(&end_timer);
	seconds = difftime(start_timer, end_timer);

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
		plot_setting.plot_2D_interval_MATLAB("./outputs/", "B4", result.tmv_flowpipes, setting);
	}
	cout << r_loss << endl;
	return 0;
}