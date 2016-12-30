//
// Created by Mingkai Chen on 2016-12-27.
//

#include "graph/mutable/mutable_connector.hpp"
#include "graph/operation/elementary.hpp"
#include "graph/operation/transform.hpp"
#include "graph/operation/matmul.hpp"
#include "graph/variable/placeholder.hpp"
#include "graph/variable/variable.hpp"
#include "creator_vertex.hpp"

#ifdef creator_vertex_hpp

namespace tensorio
{

// adhoc function mapping connector types to connector constructors
static nnet::mutable_connector<double>* mutable_build (CONNECTOR_TYPE cm)
{
	size_t nargs = 0;
	nnet::MAKE_CONNECT<double> maker;
	switch (cm)
	{
		case ABS:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return +args[0];
			};
			break;
		case NEG:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return -args[0];
			};
			break;
		case SIN:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::sin(args[0]);
			};
			break;
		case COS:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::cos(args[0]);
			};
			break;
		case TAN:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::tan(args[0]);
			};
			break;
		case CSC:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::csc(args[0]);
			};
			break;
		case SEC:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::sec(args[0]);
			};
			break;
		case COT:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::cot(args[0]);
			};
			break;
		case EXP:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::exp(args[0]);
			};
			break;
		case ADD:
			nargs = 2;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return args[0] + args[1];
			};
			break;
		case SUB:
			nargs = 2;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return args[0] - args[1];
			};
			break;
		case MUL:
			nargs = 2;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return args[0] * args[1];
			};
			break;
		case DIV:
			nargs = 2;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return args[0] / args[1];
			};
			break;
		case MATMUL:
			nargs = 2;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::matmul<double>::build(args[0], args[1]);
			};
			break;
		case FIT:
			nargs = 2;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::fit(args[0], args[1]);
			};
			break;
		case TRANS:
			nargs = 1;
			maker = [](std::vector<nnet::varptr<double> >& args) -> nnet::ivariable<double>*
			{
				return nnet::transpose(args[0]);
			};
			break;
		default:
			return nullptr;
	}
	return nnet::mutable_connector<double>::build(maker, nargs);
}

struct vertex_manager::node_registry
{
	// pairs metainfo and ivariable pointer
	struct info_wrapper
	{
		nnet::ivariable<double>* ptr_;
		metainfo info_;
	};

	std::unordered_map<std::string,info_wrapper> nodes_;

	// delete all ptr_s in node_ registry
	~node_registry (void)
	{
		std::unordered_set<nnet::ivariable<double>*> indeps;
		for (auto ns : nodes_)
		{
			// only mark independent nodes and leaves for deletion
			if (ns.second.info_.op_type_)
			{
				nnet::mutable_connector<double>* mc = dynamic_cast<nnet::mutable_connector<double>*>(ns.second.ptr_);
				// invalid arguments means mutable connector is not dependent (mark to destroy)
				if (false == mc->valid_args())
				{
					indeps.emplace(ns.second.ptr_);
				}
			}
			// ns.second.ptr_ is a leaf (mark to destroy)
			else
			{
				indeps.emplace(ns.second.ptr_);
			}
		}
		// destroy
		for (nnet::ivariable<double>* v : indeps)
		{
			delete v;
		}
	}

	// check if id exists, extract the node, then return its type
	bool grab_node (info_wrapper& out, std::string id)
	{
		auto it = nodes_.find(id);
		// return if id doesn't exist in repository
		if (nodes_.end() == it)
		{
			return false;
		}
		out = it->second;
		return true;
	}
};

vertex_manager::vertex_manager (void) : inst(new node_registry()) {}

vertex_manager::~vertex_manager (void) { delete inst; }

std::string vertex_manager::register_op (CONNECTOR_TYPE cm)
{
	node_registry::info_wrapper wrap;
	nnet::mutable_connector<double>* con = mutable_build(cm);
	std::string id = con->get_uid(); // get uid
	wrap.ptr_ = con;
	wrap.info_.op_type_ = cm; // store connector type
	inst->nodes_[id] = wrap;
	return id;
}

std::string vertex_manager::register_leaf (std::string label, var_opt opt)
{
	nnet::ileaf<double>* leaf = nullptr;
	switch (opt.type)
	{
		case PLACE:
			leaf = new nnet::placeholder<double>(opt.shape_, label);
			break;
		case CONST:
			{
				nnet::const_init<double> init(opt.parameter_->val_);
				leaf = new nnet::variable<double>(opt.shape_, init, label);
			}
			break;
		case RAND:
			{
				double min = opt.parameter_->min2max_.first;
				double max = opt.parameter_->min2max_.second;
				nnet::random_uniform<double> init(min, max);
				leaf = new nnet::variable<double>(opt.shape_, init, label);
			}
			break;
		default:
			// error
			throw std::invalid_argument("poorly defined variable option");
	}
	std::string id = leaf->get_uid();
	// store data in registry
	node_registry::info_wrapper wrap;
	wrap.ptr_ = leaf;
	inst->nodes_[id] = wrap;
	return id;
}

bool vertex_manager::delete_node (std::string id)
{
	node_registry::info_wrapper wrapper;
	if (inst->grab_node(wrapper, id))
	{
		inst->nodes_.erase(id);
		// unattach permanent node from mutable_connector wrapper
		ccoms::update_message msg;
		msg.cmd_ = ccoms::update_message::REMOVE_ARG;
		wrapper.ptr_->notify(msg);
		// delete wrapper.ptr_: triggering chain destruction
		delete wrapper.ptr_;
		return true;
	}
	return false;
}

void vertex_manager::link (std::string id1, std::string id2, size_t index)
{
	node_registry::info_wrapper w1;
	node_registry::info_wrapper w2;

	bool g1 = inst->grab_node(w1, id1);
	// error: one non-existent node
	if (false == g1 ||
		false == inst->grab_node(w2, id2))
	{
		std::string invalid_id = g1 ? id1 : id2;
		throw std::invalid_argument("invalid id: (" + invalid_id + ") node not found");
	}

	// error: can't connect to a leaf
	if (!w2.info_.op_type_)
	{
		throw std::invalid_argument("can't connect to leaf " + w2.ptr_->get_name());
	}

	nnet::mutable_connector<double>* mc =
			static_cast<nnet::mutable_connector<double>*>(w2.ptr_);
	mc->add_arg(w1.ptr_, index);
}

bool vertex_manager::delete_link (std::string id, size_t index)
{
	node_registry::info_wrapper wrap;
	// error: one non-existent node
	if (false == inst->grab_node(wrap, id) || !wrap.info_.op_type_)
	{
		return false;
	}

	static_cast<nnet::mutable_connector<double>*>(wrap.ptr_)->remove_arg(index);
	return true;
}

std::experimental::optional<metainfo> vertex_manager::node_info (std::string id)
{
	std::experimental::optional<metainfo> info;
	node_registry::info_wrapper wrap;
	if (inst->grab_node(wrap, id))
	{
		info = wrap.info_;
	}
	return info;
}

void vertex_manager::get_connections (std::vector<connection>& conns, std::string root_id)
{
}

void vertex_manager::get_forwards (std::vector<std::string>& ids)
{
}

void vertex_manager::get_backwards (std::vector<std::string>& ids)
{
}

}

#endif