#include "../flowstar/flowstar-toolbox/Continuous.h"

using namespace std;
using namespace flowstar;

string sigmoid(const string &s1, const string &s2, const string &s3, const string s4, const string &s5, const string &s6, const string &s7) {
	return "1.0 / (1 + exp(-1 * ( x1 * (" + s1 + ") + x2 * (" + s2 + ") + x3 * (" + s3 + ") + x4 * (" + s4 + ") + x5 * (" + s5 + ") + x6 * (" + s6 + ") + (" + s7 + "))))";
}

string iteconstant(const string &sig_str, const string &branch1, const string &branch2) {
	return "(" + sig_str + ") * (" + branch1 + ") + (1 - (" + sig_str + ")) * (" + branch2 + ")";
}

string nested2iteconstant(
	const string &s1, const string &s2, const string &s3, const string &s4, const string &s5, const string &s6, const string &s7, const string &s8,
	const string &s9, const string &s10, const string &s11, const string &s12, const string &s13, const string &s14, const string &s15, const string &s16,
	const string &s17, const string &s18, const string &s19, const string &s20, const string &s21, const string &s22, const string &s23, const string &s24, const string &s25
) {
	// most inner branch
	string inner_sig = sigmoid(s17, s18, s19, s20, s21, s22, s23);
	string inner = iteconstant(inner_sig, s24, s25);

	// middle branch
	string middle_sig = sigmoid(s9, s10, s11, s12, s13, s14, s15);
	string middle = iteconstant(middle_sig, s16, inner);

	// outer branch
	string outer_sig = sigmoid(s1, s2, s3, s4, s5, s6, s7);
	string outer = iteconstant(outer_sig, s8, middle);

	return outer;
}

double relu(double x) {
	return max(x, 0.0);
}

int main(int argc, char *argv[])
{
    
//    intervalNumPrecision = 300;
    
	// Declaration of the state variables.
	unsigned int numVars = 9;
    
    Variables vars;

	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int x4_id = vars.declareVar("x4");
	int x5_id = vars.declareVar("x5");
	int x6_id = vars.declareVar("x6");
	int u1_id = vars.declareVar("u1");
	int u2_id = vars.declareVar("u2");
	int u3_id = vars.declareVar("u3");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
    ODE<Real> dynamics({"x4", "x5 + 0.25", "x6", "9.81 * sin(u1) / cos(u1)", "-9.81 * sin(u2) / cos(u2)", "u3 - 9.81", "0", "0", "0"}, vars);
    
	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 2;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.05, order);

	// time horizon for a single control step
//	setting.setTime(0.1);

	// cutoff threshold
	setting.setCutoffThreshold(1e-7);
 
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
		vector<double> range;
	for (int i = 0; i < 12; ++i) {
		range.push_back(atof(argv[i + 34]));
	}
	Interval init_x1(range[0], range[1]), init_x2(range[2], range[3]), init_x3(range[4], range[5]), init_x4(range[6], range[7]), init_x5(range[8], range[9]), init_x6(range[10], range[11]);
	Interval init_u1(0), init_u2(0), init_u3(0);
	std::vector<Interval> X0;
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_x5);
	X0.push_back(init_x6);
	X0.push_back(init_u1);
	X0.push_back(init_u2);
	X0.push_back(init_u3);

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
	s.resize(34); // s[0] is of no use
	for (int i = 1; i <= 33; ++i) {
		s[i] = argv[i];
	}

	bool plot = false;
	if (argc == 35) {
		string plt = argv[34];
		if (plt == "--plot") {
			plot = true;
		}
	}

	double score = 0;
	for (int iter = 0; iter < 5; ++iter)
	{
		// cout << "Step " << iter << " starts.      " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		
		
		string strExpU1 = nested2iteconstant(
			s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8],
			s[11], s[12], s[13], s[14], s[15], s[16], s[17], s[18],
			s[21], s[22], s[23], s[24], s[25], s[26], s[27], s[28], s[31]
		);
		
		string strExpU2 = nested2iteconstant(
			s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[9],
			s[11], s[12], s[13], s[14], s[15], s[16], s[17], s[19],
			s[21], s[22], s[23], s[24], s[25], s[26], s[27], s[29], s[32]
		);

		string strExpU3 = nested2iteconstant(
			s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[10],
			s[11], s[12], s[13], s[14], s[15], s[16], s[17], s[20],
			s[21], s[22], s[23], s[24], s[25], s[26], s[27], s[30], s[33]
		);

		Expression<Real> exp_u1(strExpU1, vars);
		Expression<Real> exp_u2(strExpU2, vars);
		Expression<Real> exp_u3(strExpU3, vars);

        TaylorModel<Real> tm_u1;
        TaylorModel<Real> tm_u2;
        TaylorModel<Real> tm_u3;
        exp_u1.evaluate(tm_u1, initial_set.tmvPre.tms, order, initial_set.domain, setting.tm_setting.cutoff_threshold, setting.g_setting);
        exp_u2.evaluate(tm_u2, initial_set.tmvPre.tms, order, initial_set.domain, setting.tm_setting.cutoff_threshold, setting.g_setting);
        exp_u3.evaluate(tm_u3, initial_set.tmvPre.tms, order, initial_set.domain, setting.tm_setting.cutoff_threshold, setting.g_setting);
        tm_u1.remainder.bloat(0);
        tm_u2.remainder.bloat(0);
        tm_u3.remainder.bloat(0);

        initial_set.tmvPre.tms[u1_id] = tm_u1;
        initial_set.tmvPre.tms[u2_id] = tm_u2;
        initial_set.tmvPre.tms[u3_id] = tm_u3;

        dynamics.reach(result, initial_set, 0.2, setting, safeSet, symbolic_remainder);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
            vector<Interval> inter_box;
			result.fp_end_of_time.intEval(inter_box, order, setting.tm_setting.cutoff_threshold);
            double max_loss = 0.0; 
			max_loss = max(max_loss, relu(-1 * inter_box[0].inf() - 0.32));
			max_loss = max(max_loss, relu(inter_box[0].sup() - 0.32));
			max_loss = max(max_loss, relu(-1 * inter_box[1].inf() - 0.32));
			max_loss = max(max_loss, relu(inter_box[1].sup() - 0.32));
			max_loss = max(max_loss, relu(-1 * inter_box[2].inf() - 0.32));
			max_loss = max(max_loss, relu(inter_box[2].sup() - 0.32));
			score += max_loss;
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

	vector<Interval> end_box;
	result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);
	for (int i = 0; i < 6; ++i) {
		cout << fixed << end_box[i].inf() << " ";
		cout << fixed << end_box[i].sup() << " ";
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
		plot_setting.plot_2D_interval_MATLAB("./outputs/", "qmpc3", result.tmv_flowpipes, setting);
	}
	cout << score << endl;
	return 0;
}