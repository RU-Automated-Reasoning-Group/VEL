#include "../flowstar/flowstar-toolbox/Continuous.h"

using namespace std;
using namespace flowstar;

double relu(double x) {
	return max(x, 0.0);
}

double reach_loss(vector<double> reach_set, vector<double> target_set) {
    // for (int i = 0; i < reach_set.size(); ++i) {
    //     cout << reach_set[i] << " " << target_set[i] << endl;
    // }
	double val = 0;
	val = max(relu(reach_set[1] - target_set[1]), val); // x_sup
	val = max(relu(target_set[0] - reach_set[0]), val); // x_inf

	val = max(relu(reach_set[3] - target_set[3]), val); // y_sup
	val = max(relu(target_set[2] - reach_set[2]), val); // y_inf

    val = max(relu(reach_set[5] - target_set[5]), val); // x3_sup
	val = max(relu(target_set[4] - reach_set[4]), val); // x3_inf
	
    val = max(relu(reach_set[7] - target_set[7]), val); // x4_sup
	val = max(relu(target_set[6] - reach_set[6]), val); // x4_inf

	return val;
}

int main(int argc, char* argv[])
{
    unsigned int numVars = 6;
    
    Variables vars;

	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int x4_id = vars.declareVar("x4");
	int u_id = vars.declareVar("u");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
    ODE<Real> dynamics({"x2", "-x1 + 0.1 * sin(x3)", "x4", "u", "0"}, vars);
    
	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 4;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.01, order);

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
	Interval init_x1(atof(argv[6]), atof(argv[7])), init_x2(atof(argv[8]), atof(argv[9])), init_x3(-0.1, 0.1), init_x4(-0.1, 0.1);
	// Interval init_x0(-0.25 - w, -0.25 + w), init_x1(-0.25 - w, -0.25 + w), init_x2(0.35 - w, 0.35 + w), init_x3(-0.35 - w, -0.35 + w), init_x4(0.45 - w, 0.45 + w), init_x5(-0.35 - w, -0.35 + w);
	Interval init_u(0);
	std::vector<Interval> X0;
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
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
	string s5 = argv[5];

	bool plot = false;
	if (argc == 11) {
		string plt = argv[10];
		if (plt == "--plot") {
			plot = true;
		}
	}

	double score = 0;
	for (int iter = 0; iter < 16; ++iter)
	{
		// cout << "Step " << iter << " starts.      " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		
		
        string strExpU = "x1*(" + s1 + ") + x2*(" + s2 + ") + x3*(" + s3 + ") +x4*(" + s4 + ") + (" + s5 + ")";

		Expression<Real> exp_u(strExpU, vars);

        TaylorModel<Real> tm_u;
        exp_u.evaluate(tm_u, initial_set.tmvPre.tms, order, initial_set.domain, setting.tm_setting.cutoff_threshold, setting.g_setting);

        initial_set.tmvPre.tms[u_id] = tm_u;

        dynamics.reach(result, initial_set, 0.1, setting, safeSet, symbolic_remainder);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
            vector<Interval> inter_box;
			result.fp_end_of_time.intEval(inter_box, order, setting.tm_setting.cutoff_threshold);
			// for (int i = 0; i < 5; ++i) {
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
    for (int i = 0; i < 4; ++i) {
        target_set.push_back(-0.1);
	    target_set.push_back(0.1);
    }

	vector<Interval> end_box;
	result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);
	vector<double> reach_set;
    for (int i = 0; i < 4; ++i) {
        reach_set.push_back(end_box[i].inf());
        reach_set.push_back(end_box[i].sup());
    }

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
		plot_setting.plot_2D_interval_MATLAB("./outputs/", "toraeq", result.tmv_flowpipes, setting);
	}
	cout << r_loss << endl;
	return 0;
}
