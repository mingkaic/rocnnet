//
//  tensor_writer.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2017-04-25.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_WRITER_HPP

namespace nnet
{

template <typename T>
bool write_inorder (std::vector<inode<T>*>& serialvec, std::string label, std::string fname)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	tenncor::repository repo;

	{
		// Read the existing address book.
		std::fstream input(fname, std::ios::in | std::ios::binary);
		if (!input)
		{
			std::cout << fname << ": File not found. Creating a new file." << std::endl;
		}
		else if (!repo.ParseFromIstream(&input))
		{
			std::cerr << "Failed to parse tenncor repository." << std::endl;
			return false;
		}
	}

	for (size_t i = 0, n = serialvec.size(); i < n; i++)
	{
		std::string node_label = label + ":" + std::to_string(i);
		nnet::inode<T>* serialelem = serialvec[i];
		tenncor::tensor_proto proto;
		const nnet::tensor<T>* serialtens = serialelem->eval();
		if (nullptr == serialtens) continue; // we can't serialize something not initialized
		serialtens->serialize(&proto);
		(*repo.mutable_node_map())[node_label] = proto;
	}

	{
		// Write the new address book back to disk.
		std::fstream output(fname, std::ios::out | std::ios::trunc | std::ios::binary);
		if (!repo.SerializeToOstream(&output)) {
			std::cerr << "Failed to write address book." << std::endl;
			return false;
		}
	}

	google::protobuf::ShutdownProtobufLibrary();
	return true;
}

using node_map_t = google::protobuf::Map<std::string,tenncor::tensor_proto>;

template<typename T>
bool read_inorder (std::vector<inode<T>*>& deserialvec, std::string label, std::string fname)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	tenncor::repository repo;

	{
		std::fstream input(fname, std::ios::in | std::ios::binary);
		if (!repo.ParseFromIstream(&input))
		{
			return false;
		}
	}

	std::unordered_set<std::string> existing_labels;
	std::vector<inode<T>*> unfound;
	for (size_t i = 0, n = deserialvec.size(); i < n; i++)
	{
		std::string node_label = label + ":" + std::to_string(i);
		nnet::inode<T>* deserelem = deserialvec[i];

		node_map_t::const_iterator it = repo.node_map().find(node_label);
		if (repo.node_map().end() == it)
		{
			// warn
			unfound.push_back(deserelem);
		}
		else if (false == deserelem->read_proto(it->second))
		{
			// warn2
			unfound.push_back(deserelem);
		}
	}
	deserialvec = unfound;

	google::protobuf::ShutdownProtobufLibrary();

	return true;
}

}

#endif