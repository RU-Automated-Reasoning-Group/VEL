#include "../flowstar/flowstar-toolbox/Continuous.h"

using namespace std;
using namespace flowstar;

string sigmoid(const string &s1, const string &s2, const string &s3, const string s4, const string &s5) {
	return "1.0 / (1.0 + exp(-1.0 * ( x1 * (" + s1 + ") + x2 * (" + s2 + ") + x3 * (" + s3 + ") + x4 * (" + s4 + ") + (" + s5 + "))))";
}

string linear (const string &s1, const string &s2, const string &s3, const string &s4, const string &s5) {
    return "x1 * (" + s1 + ") + x2 * (" + s2 + ") + x3 * (" + s3 + ") + x4 * (" + s4 + ") + (" + s5 + ")";
}

string ite(const string &sig_str, const string &branch1, const string &branch2) {
	return "(" + sig_str + ") * (" + branch1 + ") + (1 - (" + sig_str + ")) * (" + branch2 + ")";
}

string itelinear(
    const string &s1, const string &s2, const string &s3, const string &s4, const string &s5,
    const string &s6, const string &s7, const string &s8, const string &s9, const string &s10,
    const string &s11, const string &s12, const string &s13, const string &s14, const string &s15
) {
    string sig_str = sigmoid(s1, s2, s3, s4, s5);
    string branch1 = linear(s6, s7, s8, s9, s10);
    string branch2 = linear(s11, s12, s13, s14, s15);
    return ite(sig_str, branch1, branch2);
}

double relu(double x) {
	return max(x, 0.0);
}

double reach_loss(vector<double> reach_set, vector<double> target_set) {
	double val = 0;
	val = max(relu(reach_set[1] - target_set[1]), val); // x1_sup
	val = max(relu(target_set[0] - reach_set[0]), val); // x1_inf

	val = max(relu(reach_set[3] - target_set[3]), val); // x2_sup
	val = max(relu(target_set[2] - reach_set[2]), val); // x2_inf
	
	val = max(relu(reach_set[5] - target_set[5]), val); // x3_sup
	val = max(relu(target_set[4] - reach_set[4]), val); // x3_inf
	
	val = max(relu(reach_set[7] - target_set[7]), val); // x4_sup
	val = max(relu(target_set[6] - reach_set[6]), val); // x4_inf

	return val;
}

int main(int argc, char *argv[])
{
    
//    intervalNumPrecision = 300;
    
	// Declaration of the state variables.
	unsigned int numVars = 6;
    
    Variables vars;

	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int x4_id = vars.declareVar("x4");
	int u1_id = vars.declareVar("u1");
	int u2_id = vars.declareVar("u2");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
    ODE<Real> dynamics({"x4 * cos(x3)", "x4 * sin(x3)", "u2", "u1", "0", "0"}, vars);
    
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
	Interval init_x1(9.5, 9.55), init_x2(-4.5, -4.45), init_x3(2.1, 2.11), init_x4(1.5, 1.51);
	// Interval init_x0(-0.25 - w, -0.25 + w), init_x1(-0.25 - w, -0.25 + w), init_x2(0.35 - w, 0.35 + w), init_x3(-0.35 - w, -0.35 + w), init_x4(0.45 - w, 0.45 + w), init_x5(-0.35 - w, -0.35 + w);
	Interval init_u1(0), init_u2(0);
	std::vector<Interval> X0;
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_u1);
	X0.push_back(init_u2);

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

	vector<string> s;
    s.resize(26);
    for (int i = 1; i <= 25; ++i) {
        s[i] = argv[i];
    }

	bool plot = false;
	if (argc == 27) {
		string plt = argv[26];
		if (plt == "--plot") {
			plot = true;
		}
	}

	double score = 0;
	for (int iter = 0; iter < 30; ++iter)
	{
		// cout << "Step " << iter << " starts.      " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		
		
		string strExpU1 = itelinear(
                s[1], s[2], s[3], s[4], s[5],
                s[6], s[7], s[8], s[9], s[14],
                s[16], s[17], s[18], s[19], s[24]
            );
        // string strExpU1 = "exp(-1.0 * ( x1 * (" + s[1] + ") + x2 * (" + s[2] + ") + x3 * (" + s[3] + ") + x4 * (" + s[4] + ") + (" + s[5] + ")))";

        string strExpU2 = itelinear(
                s[1], s[2], s[3], s[4], s[5],
                s[10], s[11], s[12], s[13], s[15],
                s[20], s[21], s[22], s[23], s[25]
            );

		Expression<Real> exp_u1(strExpU1, vars);
		Expression<Real> exp_u2(strExpU2, vars);

        TaylorModel<Real> tm_u1;
        TaylorModel<Real> tm_u2;
        exp_u1.evaluate(tm_u1, initial_set.tmvPre.tms, order, initial_set.domain, setting.tm_setting.cutoff_threshold, setting.g_setting);
        exp_u2.evaluate(tm_u2, initial_set.tmvPre.tms, order, initial_set.domain, setting.tm_setting.cutoff_threshold, setting.g_setting);
        tm_u1.remainder.bloat(1e-4);
        tm_u2.remainder.bloat(0);

        initial_set.tmvPre.tms[u1_id] = tm_u1;
        initial_set.tmvPre.tms[u2_id] = tm_u2;

        dynamics.reach(result, initial_set, 0.2, setting, safeSet, symbolic_remainder);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
            vector<Interval> inter_box;
			result.fp_end_of_time.intEval(inter_box, order, setting.tm_setting.cutoff_threshold);
			// for (int i = 0; i < 6; ++i) {
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
	target_set.push_back(-0.6);
	target_set.push_back(0.6);
	target_set.push_back(-0.2);
	target_set.push_back(0.2);
	target_set.push_back(-0.06);
	target_set.push_back(0.06);
	target_set.push_back(-0.3);
	target_set.push_back(0.3);

	vector<Interval> end_box;
	result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);
	vector<double> reach_set;
	reach_set.push_back(end_box[0].inf());
	reach_set.push_back(end_box[0].sup());
	reach_set.push_back(end_box[1].inf());
	reach_set.push_back(end_box[1].sup());
	reach_set.push_back(end_box[2].inf());
	reach_set.push_back(end_box[2].sup());
	reach_set.push_back(end_box[3].inf());
	reach_set.push_back(end_box[3].sup());

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
		plot_setting.plot_2D_interval_MATLAB("./outputs/", "unicycle_car", result.tmv_flowpipes, setting);
	}
	cout << r_loss << endl;
	return 0;
}