#include <fstream>

#include "perf/measure.hpp"

#ifdef PERF_MEASURE_HPP

namespace perf
{

PerfRecord& get_global_record (void)
{
	static PerfRecord record(
		[](PerfRecord& deletion)
		{
			if (false == deletion.empty())
			{
				std::ofstream outf("/tmp/performance.csv");
				deletion.to_csv(outf);
				outf.flush();
			}
		});
	return record;
}

}

#endif
