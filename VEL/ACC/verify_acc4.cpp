#include "../flowstar/flowstar-toolbox/Continuous.h"

using namespace std;
using namespace flowstar;

string sigmoid(const string &s1, const string &s2, const string &s3, const string s4, const string &s5, const string &s6) {
	return "1.0 / (1 + exp(-1 * ( 30 * (" + s1 + ") + 1.4 * (" + s2 + ") + x5 * (" + s3 + ") + (x1-x4) * (" + s4 + ") + (x2-x5)*(" + s5 + ") + (" + s6 + "))))";
}

string linear(const string &s1, const string &s2, const string &s3, const string &s4, const string &s5, const string &s6) {
    return "30*(" + s1 + ") + 1.4*(" + s2 + ") + x5*(" + s3 + ") + (x1-x4)*(" + s4 + ") + (x2-x5)*(" + s5 + ") + (" + s6 + ")";
}

double relu(double x) {
	return max(x, 0.0);
}

double loss(vector<double> reach_set, vector<double> target_set) {
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
	unsigned int numVars = 7;
    
    Variables vars;

	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int x4_id = vars.declareVar("x4");
	int x5_id = vars.declareVar("x5");
	int x6_id = vars.declareVar("x6");
	int u_id = vars.declareVar("u");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
    ODE<Real> dynamics({"x2", "x3", "-4 - 2*x3 - 0.0001*x2*x2", "x5", "x6", "2*u - 2*x6 - 0.0001*x5*x5", "0"}, vars);
    
	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 2;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.02, order);

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
	Interval init_x1(105.0, 110.0), init_x2(32.0, 32.2), init_x3(0.0, 0.0), init_x4(10.0, 11.0), init_x5(30.0, 30.2), init_x6(0.0, 0.0);
	// Interval init_x0(-0.25 - w, -0.25 + w), init_x1(-0.25 - w, -0.25 + w), init_x2(0.35 - w, 0.35 + w), init_x3(-0.35 - w, -0.35 + w), init_x4(0.45 - w, 0.45 + w), init_x5(-0.35 - w, -0.35 + w);
	Interval init_u(0);
	std::vector<Interval> X0;
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_x5);
	X0.push_back(init_x6);
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
	string s6 = argv[6];
    string s7 = argv[7];
    string s8 = argv[8];
    string s9 = argv[9];
    string s10 = argv[10];
    string s11 = argv[11];
    string s12 = argv[12];
    string s13 = argv[13];
    string s14 = argv[14];
    string s15 = argv[15];
    string s16 = argv[16];
    string s17 = argv[17];
    string s18 = argv[18];

	bool plot = false;
	if (argc == 20) {
		string plt = argv[19];
		if (plt == "--plot") {
			plot = true;
		}
	}

	double score = 0;
	for (int iter = 0; iter < 50; ++iter)
	{
		// cout << "Step " << iter << " starts.      " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		
		
		string sig_str = sigmoid(s1, s2, s3, s4, s5, s6);
        string branch1 = linear(s7, s8, s9, s10, s11, s12);
        string branch2 = linear(s13, s14, s15, s16, s17, s18);
		string strExpU = "(" + sig_str + ") * (" + branch1 + ") + (1 - (" + sig_str + ")) * (" + branch2 + ")";
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
			double p1_loss = relu(-1 * inter_box[0].inf() + inter_box[3].sup() + 1.4 * inter_box[4].sup() + 10);
			double p2_loss = relu(inter_box[0].sup() - inter_box[3].inf() - 102);
			score += max(p1_loss, p2_loss);// for (int i = 0; i < 6; ++i) {
            // for (int i = 0; i < 7; ++i) {
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
		plot_setting.plot_2D_interval_MATLAB("./outputs/", "acc1", result.tmv_flowpipes, setting);
	}
	cout << score << endl;
	return 0;
}