#include "fuzz.h"

#ifdef TENNCOR_FUZZ_H

namespace FUZZ
{

static std::default_random_engine generator;

void delim (void)
{
    fuzzLogger.close();
	remove(FUZZ_FILE);
    fuzzLogger.open(FUZZ_FILE);
}

std::vector<double> getDouble (size_t len,
	std::string purpose,
    std::pair<double,double> range)
{
	double min, max;
	if (range.first == range.second)
	{
		min = std::numeric_limits<double>::min();
		max = std::numeric_limits<double>::max();
	}
	else
	{
		min = range.first;
		max = range.second;
	}

	std::vector<double> vec;
	std::uniform_real_distribution<double> dis(min, max);

	fuzzLogger << purpose << ": double<";
	for (size_t i = 0; i < len; i++)
	{
		double val = dis(generator);
		fuzzLogger << val<< ",";
		vec.push_back(val);
	}
	fuzzLogger << ">" << std::endl;

	return vec;
}

std::vector<size_t> getInt (size_t len,
	std::string purpose,
    std::pair<size_t,size_t> range)
{
	size_t min, max;
	if (range.first == range.second)
	{
		min = std::numeric_limits<size_t>::min();
		max = std::numeric_limits<size_t>::max();
	}
	else
	{
		min = range.first;
		max = range.second;
	}

	std::vector<size_t> vec;
	fuzzLogger << purpose << ": int<";
	std::uniform_int_distribution<size_t> dis(min, max);
	for (size_t i = 0; i < len; i++)
	{
		size_t val = dis(generator);
		fuzzLogger << val << ",";
		vec.push_back(val);
	}
	fuzzLogger << ">" << std::endl;

	return vec;
}

std::string getString (size_t len,
	std::string purpose,
    std::string alphanum)
{
	std::vector<size_t> indices = FUZZ::getInt(len, "indices", {0, alphanum.size()-1});
	std::string s(len, ' ');
	std::transform(indices.begin(), indices.end(), s.begin(),
	[&alphanum](size_t index)
	{
		return alphanum[index];
	});
	fuzzLogger << purpose << ": string<" << s << ">" << std::endl;

	return s;
}

}

#endif