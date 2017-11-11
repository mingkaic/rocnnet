//
// Created by Mingkai Chen on 2017-01-30.
//

#include "edgeinfo/csv_record.hpp"

#ifdef csv_record_hpp

namespace rocnnet_record
{

#ifdef CSV_RCD

std::unique_ptr<igraph_record> record_status::rec =
	std::make_unique<csv_record>("op-profile.csv");
bool record_status::rec_good = true;

#endif /* CSV_RCD */

csv_record::csv_record (std::string fname) :
	outname_(fname) {}

void csv_record::setVerbose (bool verbosity)
{
	verbose_ = verbosity;
}

void csv_record::setDisplayShape (bool display)
{
	display_shape_ = display;
}

}

#endif