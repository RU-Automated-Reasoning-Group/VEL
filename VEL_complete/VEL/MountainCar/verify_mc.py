import subprocess
import multiprocess as mp
import sys
import numpy as np
flowstar_path = "../flowstar/flowstar-2.1.0/flowstar-2.1.0/flowstar"

def _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts):

    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
    parallel_runs = [pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list]
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts-1)
    pool.close()
    pool.terminate()
    pool.join()
    return results

def eval_one_interval(index, interval, params):
    model_str = """hybrid reachability
{
	state var temp, u, pos, vel, k, t

	setting
	{
		fixed steps 0.004
		time 1000
		remainder estimation 1e-6
		identity precondition 
		gnuplot interval pos, vel
		adaptive orders { min 4 , max 100 }
		cutoff 1e-10
		precision 200
		output mctestgnu%d
		max jumps 9000
		print off
	}

	modes
	{
        m3
        {
            nonpoly ode
            {
                temp' = 0.0
                u' = 0.0
                pos' = 0.0
                vel' = 0.0
                t' = 1.0
                k' = 0.0
            }
            inv
            {
                t <= 0.0
            }        
        }
        m0
		{
			nonpoly ode
			{
                temp' = 0.0
                u' = 0.0
                pos' = 0.0
                vel' = 0.0
                t' = 1.0
                k' = 0.0
			}
			inv
			{
                t <= 0.0
			}
		}
        m1
        {
            nonpoly ode
            {
                temp' = 0.0
                u' = (%f) * (1.0 / (1.0 + exp((-1.0) * (pos * (%f) + vel * (%f) + (%f))))) + (%f) * (1.0 - (1.0 / (1.0 + exp((-1.0) * (pos * (%f) + vel * (%f) + (%f))))))
                pos' = 0.0
                vel' = 0.0
                t' = 1.0
                k' = 0.0
            }
            inv
            {
                t <= 1.0
            }
        }
        m2
        {
            nonpoly ode
            {
                temp' = 0.0015 * u - 0.0025 * cos(3.0 * pos)
                pos' = 0.0
                vel' = 0.0
                u' = 0.0
                t' = 1.0
                k' = 0.0
            }
            inv
            {
                t <= 1.0
            }
        }
        m4
        {
            nonpoly ode
            {
                temp' = 0.0
                pos' = 0.0
                vel' = 0.0
                u' = 0.0
                t' = 1.0
                k' = 0.0
            }
            inv
            {
                t <= 0.0
            }
        }
        unsafe1
        {
            nonpoly ode
            {
                temp' = 0.0
                pos' = 0.0
                vel' = 0.0
                u' = 0.0
                t' = 1.0
                k' = 0.0
            }
            inv
            {
                t <= 1.0
            }
        }
        speed
        {
            nonpoly ode
            {
                temp' = 0.0
                pos' = 0.0
                vel' = 0.0
                u' = 0.0
                k' = 0.0
                t' = 1.0
            }
            inv
            {
                t <= 1.0
            }
        }
	}
	jumps
	{
        m3 -> m0
        guard { t = 0.0 pos >= -1.2 pos <= 0.45 }
        reset { t' := 0.0 }
        parallelotope aggregation { }
        m3 -> m0
        guard { t = 0.0 pos <= -1.2 }
        reset { t' := 0.0 pos' := -1.2 vel' := 0.0}
        parallelotope aggregation { }
        m0 -> m1
        guard { }
        reset { t' := 0.0 u' := 0 k' := k + 1.0}
        parallelotope aggregation { }
        m1 -> m2
        guard { u >= 1.0 t = 1.0}
        reset { u' := 1.0 t' := 0.0 temp' := 0.0}
        parallelotope aggregation { }
        m1 -> m2
        guard { u <= -1.0 t = 1.0}
        reset { u' := -1.0 t' := 0.0 temp' := 0.0}
        parallelotope aggregation { }
        m1 -> m2
        guard { u >= -1.0 u <= 1.0 t = 1.0 }
        reset { t' := 0.0 temp' := 0.0}
        parallelotope aggregation { }
        m2 -> m3
        guard { t = 1.0 vel + temp <= 0.07 vel + temp >= -0.07 }
        reset { t' := 0.0 pos' := pos + vel + temp vel' := vel + temp }
        parallelotope aggregation { }
        m2 -> m3
        guard { t = 1.0 vel + temp >= 0.07 }
        reset { pos' := pos + 0.07 vel' := 0.07 t' := 0.0 }
        parallelotope aggregation { }
        m2 -> m3
        guard { t = 1.0 vel + temp <= -0.07 }
        reset { pos' := pos + -0.07 vel' := -0.07 t' := 0.0 }
        parallelotope aggregation { }
        m3 -> m4
        guard { t = 0.0 pos >= 0.45 }
        reset { }
        parallelotope aggregation { }
        m3 -> unsafe1
        guard { k >= 150.0 pos <= 0.45 }
        reset { }
        interval aggregation
        m3 -> speed
        guard { pos >= 0.15 pos <= 0.25 vel <= 0.02}
        reset { }
        interval aggregation
	}
	init
	{
		m3
		{
			pos in [%.3f, %.3f]
			vel in [0, 0]
			u in [0, 0]
			temp in [0, 0]
			t in [0, 0]
            k in [0, 0]
		}
	}
}""" % (index, params[3], params[0], params[1], params[2], params[4], params[0], params[1], params[2], interval[0], interval[1])
    file_name = f"mc{index}.model"
    with open(file_name, "w") as f:
        f.write(model_str)

    cmd_str = f"{flowstar_path} < {file_name}"
    print(cmd_str)
    x = subprocess.run(cmd_str, shell=True)

    output_file = f"mctestgnu{index}.flow"
    with open("outputs/" + output_file, "r") as output_file:
        all_lines = output_file.readlines()
        count = 0
        for idx, s in enumerate(all_lines):
            if "speed" in s:
                count += 1
        if count != 1:
            return "fail"

        count = 0
        for idx, s in enumerate(all_lines):
            if "unsafe1" in s:
                count += 1
        if count != 1:
            return "fail"

        line_num = -1
        for idx, s in enumerate(all_lines):
            if s.startswith("computation paths"):
                line_num = idx
                break
        
        assert line_num > 0
        path_count = 0
        valid_path_count = 0
        begin = False
        for n in range(line_num + 1, len(all_lines) + 1):
            if all_lines[n].startswith("{"):
                begin = True
            elif all_lines[n].startswith("m3"):
                path_count += 1
                if all_lines[n].endswith("m4;\n"):
                    valid_path_count += 1
            elif begin and all_lines[n].startswith("}"):
                break
        
        return (path_count, valid_path_count)
                

    # return "succeed"





def verify_mc_full_range(params):
    intervals = [[-0.44, -0.43], [-0.43, -0.42], [-0.42, -0.415], [-0.415, -0.41], [-0.41, -0.40], [-0.6, -0.59], [-0.59, -0.58], [-0.58, -0.57], [-0.57, -0.56], [-0.56, -0.55], [-0.55, -0.54], [-0.54, -0.53], [-0.53, -0.52], [-0.52, -0.51], [-0.51, -0.5], [-0.5, -0.49], [-0.49, -0.48], [-0.48, -0.47], [-0.47, -0.46], [-0.46, -0.45], [-0.45, -0.44]]
    # print(intervals)
    # print(len(intervals))
    # exit()
    input_dict_list = []
    for index, interval in enumerate(intervals):
        print(index, interval)
        input_dict_list.append({"index": index, "interval": interval, "params": params})
    results = _try_multiprocess(eval_one_interval, input_dict_list, 8, 6000, 6000)
    print(results)
    for index, rst in enumerate(results):
        if rst == "fail":
            print("fail on interval ", index)
            continue
        path_count, valid_count = rst
        print(f"interval {index} has {path_count} paths and {valid_count} valid paths")
        if path_count != valid_count:
            print(f"interval {index} fails")
        else:
            print(f"interval {index} succeeds")


if __name__ == "__main__":
    model_file = str(sys.argv[1])
    file = open(model_file, 'r')
    lines = file.readlines()
    params = lines[0].split()
    params = [float(x) for x in params]
    params = np.array(params)
    file.close()
    verify_mc_full_range(params)
