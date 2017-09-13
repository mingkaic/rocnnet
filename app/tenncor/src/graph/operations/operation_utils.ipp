//
//  operation_utils.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2017-09-07.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_OPERATION_UTILS_HPP

namespace nnet
{

template <typename T>
inode<T>* unary_parent_search (inode<T>* operand, std::string opname)
{
	std::unordered_set<inode<T>*> audience;
	if (operand->find_audience(opname, audience))
	{
		return *audience.begin();
	}
	return nullptr;
}

template <typename T>
inode<T>* ordered_binary_parent_search (inode<T>* a, inode<T>* b, std::string opname)
{
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		// linear search on audience
		for (inode<T>* aud : audience)
		{
			std::vector<inode<T>*> args = aud->get_arguments();
			if (args.size() == 2 && args[0] == a && args[1] == b)
			{
				return aud;
			}
		}
	}
	return nullptr;
}

template <typename T>
inode<T>* unordered_binary_parent_search (inode<T>* a, inode<T>* b, std::string opname)
{
	std::unordered_set<inode<T>*> audience;
	if (a->find_audience(opname, audience))
	{
		// linear search on audience
		for (inode<T>* aud : audience)
		{
			std::vector<inode<T>*> args = aud->get_arguments();
			if (args.size() == 2 && (
					(args[0] == a && args[1] == b) ||
					(args[1] == a && args[0] == b)))
			{
				return aud;
			}
		}
	}
	return nullptr;
}

}

#endif
